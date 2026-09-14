import smtplib
import ssl
from email.message import EmailMessage

from activetigger.config import config
from activetigger.datamodels import MessagesOutModel
from activetigger.db.manager import DatabaseManager
from activetigger.errors import InvalidInputError


class Messages:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    mail_available: bool = config.mail_available
    mail_server: str | None = config.mail_server
    mail_server_port: int = config.mail_server_port
    mail_account: str | None = config.mail_account
    mail_password: str | None = config.mail_password

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        if self.mail_available:
            print("Mail service is available")
        else:
            print("Mail service is not available")

    def send_mail(self, to: str, subject: str, body: str):
        """
        Send a mail
        """
        if self.mail_server is None:
            raise Exception("Mail server is not configured")
        if self.mail_account is None:
            raise Exception("Mail account is not configured")
        if self.mail_password is None:
            raise Exception("Mail password is not configured")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Active Tigger <{self.mail_account}>"
        msg["To"] = to
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.mail_server, self.mail_server_port, context=context) as server:
            server.login(self.mail_account, self.mail_password)
            result = server.send_message(msg)

        if result != {}:
            print(f"Failed to send email to {to}: {result}")

    def send_mail_reset_password(self, user_name: str, mail: str, new_password: str) -> None:
        """
        Send a mail to reset the password
        """
        if not self.mail_available:
            raise Exception("Mail service is not available")

        subject = "Active Tigger - Password Reset"
        body = f"""
        Hello,

        Your password has been reset for the account : {user_name}
        
        Your new password is: {new_password}

        Please log in and change your password as soon as possible.

        Best regards,
        The Active Tigger Team
        """
        self.send_mail(mail, subject, body)

    @staticmethod
    def _to_out(m) -> MessagesOutModel:
        return MessagesOutModel(
            id=m.id,
            kind=m.kind,
            created_by=m.created_by,
            time=str(m.time),
            content=m.content,
            for_project=m.for_project,
            for_user=m.for_user,
        )

    def get_messages_system(self, from_user: str | None = None) -> list[MessagesOutModel]:
        """
        Get all system messages ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_system(from_user)
        return [self._to_out(m) for m in r]

    def get_messages_for_project(self, project_slug: str) -> list[MessagesOutModel]:
        """
        Get all project messages for a specific project ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_for_project(project_slug)
        return [self._to_out(m) for m in r]

    def get_messages_for_user(self, user_name: str) -> list[MessagesOutModel]:
        """
        Get all user messages for a specific user ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_for_user(user_name)
        return [self._to_out(m) for m in r]

    def get_inbox(self, user_name: str) -> list[MessagesOutModel]:
        """
        Inbox view: DMs + project-distribution copies addressed to user_name.
        """
        r = self.db_manager.messages_service.get_inbox_for_user(user_name)
        return [self._to_out(m) for m in r]

    def get_codebook_messages(self, user_name: str, project_slug: str) -> list[MessagesOutModel]:
        """
        Project-distribution copies still owned by user_name for a project.
        """
        r = self.db_manager.messages_service.get_codebook_messages(user_name, project_slug)
        return [self._to_out(m) for m in r]

    def get_messages(
        self,
        kind: str,
        from_user: str | None = None,
        for_user: str | None = None,
        for_project: str | None = None,
    ) -> list[MessagesOutModel]:
        """
        Get messages
        """
        if kind == "system":
            return self.get_messages_system(from_user)
        if kind == "project":
            if not for_project:
                raise InvalidInputError("for_project is required for kind='project'")
            return self.get_messages_for_project(for_project)
        if kind == "user":
            if not for_user:
                raise InvalidInputError("for_user is required for kind='user'")
            return self.get_messages_for_user(for_user)
        raise InvalidInputError(f"Unknown message kind: {kind}")

    def add_message(
        self, user_name: str, kind: str, content: str, property: dict | None = None
    ) -> None:
        """
        Add a message
        """
        self.db_manager.messages_service.add_message(
            user_name=user_name, kind=kind, property=property, content=content
        )

    def add_messages_bulk(
        self,
        user_name: str,
        kind: str,
        content: str,
        recipients: list[str],
        for_project: str | None = None,
    ) -> int:
        """
        Fanout: insert one row per recipient (DM = single-recipient case).
        """
        return self.db_manager.messages_service.add_messages_bulk(
            user_name=user_name,
            kind=kind,
            content=content,
            recipients=recipients,
            for_project=for_project,
        )

    def delete_message(self, id: int) -> None:
        """
        Delete a message by its ID (admin path).
        """
        self.db_manager.messages_service.delete_message(id)

    def delete_message_for_user(self, id: int, user_name: str) -> bool:
        """
        Delete a message only if it was addressed to user_name.
        Returns False if no matching row exists.
        """
        return self.db_manager.messages_service.delete_message_for_user(id, user_name)
