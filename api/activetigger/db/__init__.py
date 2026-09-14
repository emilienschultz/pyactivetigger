"""
In order to enforce the single-responsability rule, the models definitions are in db.models.py and CRUD functions are splitted into the designated object service (i.e. users operations are in users.py)

Rules of thumb:
    * Models are in db.models.py, this is the mapping between ORM entities and SQL tables
    * If two entities are linked, the relationship should be declared into the model definition
    * Each "domain" should have a service class that encapsulate the CRUD operation

Checklist:
    * Split ProjectsService.py: 1 file per entity
    * Use with keyword for session management (see UsersService.py for reference)
    * Use SQLAlchemy 2.0 API as often as possible

SQLAlchemy 2.0 recipes:
    Every SQL operation should be encapsulated into a session. Session should be handle with a "context manager", aka the `with` keyword.
    ```python
        with Session() as session:
            session.execute(...)

        print("session is automatically closed outside of the with block")

        with Session.begin() as session:
            session.execute(update(...)) # change/add something

        print("session is automatically committed and closed thanks to .begin()")
    ```
    The `Query` object is now in a [legacy state](https://docs.sqlalchemy.org/en/20/orm/queryguide/query.html) since 2.0. The recommanded method is now to use designated function like `select()`, `update()` and such.
    ```python
        with Session() as session:
            same_name_users = session.scalars(select(Users).filter_by(name=username)).all()
    ```
"""

from activetigger.errors import APIError


class DBException(APIError):
    # raise sites are "X not found" lookups, so a missing row answers 404
    status_code = 404


class DBAlreadyExists(DBException):
    status_code = 409
