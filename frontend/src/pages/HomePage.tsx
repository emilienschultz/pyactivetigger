import { marked } from 'marked';
import { FC, useMemo } from 'react';
import { FaCut, FaGithub } from 'react-icons/fa';
import { IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import logo from '../assets/at.png';
import { LoginForm } from '../components/forms/LoginForm';
import Notifications from '../components/layout/Notifications';
import { useGetActiveUsers, useGetServer } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { useAuth } from '../core/useAuth';
import { LoginParams } from '../types';

const MessageCard: FC<{ content: string; time?: string }> = ({ content, time }) => {
  const html = useMemo(() => {
    marked.setOptions({ breaks: true });
    return marked.parse(content) as string;
  }, [content]);

  return (
    <div
      style={{
        background: '#f8f9fa',
        border: '1px solid #e9ecef',
        borderLeft: '4px solid #ff9a3c',
        borderRadius: '0.5rem',
        padding: '0.75rem 1rem',
        marginBottom: '0.75rem',
        textAlign: 'left',
      }}
    >
      <div
        className="message-content"
        style={{ fontSize: '0.9rem', color: '#333', lineHeight: '1.5' }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {time && (
        <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
          {new Date(time).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export const HomePage: FC = () => {
  const { authenticatedUser, logout } = useAuth();
  const { n_users } = useGetActiveUsers();

  const {
    appContext: { developmentMode },
    setAppContext,
  } = useAppContext();

  const toggleDevelopmentMode = () => {
    setAppContext((prev) => ({ ...prev, developmentMode: !prev.developmentMode }));
  };

  const { version, mode, messages } = useGetServer(null);

  // possibility to log directly from the URL
  const navigate = useNavigate();
  const params = new URLSearchParams(window.location.search);
  const { login } = useAuth();
  if (params.get('username') && params.get('password')) {
    login({
      username: params.get('username'),
      password: params.get('password'),
    } as LoginParams).then(() => {
      navigate('/projects');
    });
  }

  return (
    <>
      <main className="container-fluid">
        <div className="row">
          {mode === 'dev' && (
            <center>
              <div className="alert alert-info mt-3">
                ℹ️ This is a <strong>development</strong> instance. Please save your data regularly.
              </div>
            </center>
          )}
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <center>
              <div className="d-flex align-items-center justify-content-center gap-3 my-3">
                <img src={logo} alt="ActiveTigger" style={{ width: '80px', height: '80px' }} />
                <div className="text-start">
                  <h1 className="mb-0 fs-2 activetigger" style={{ color: '#ff9a3c' }}>
                    Active Tigger
                  </h1>
                  <h3 className="m-0 fs-5 text-muted fw-normal">Explore & Annotate Text</h3>
                </div>
              </div>
              {n_users == null && (
                <div className="alert alert-danger mt-3">
                  ⚠️ Can't reach the API (either temporary lag or server down). You can reach out on
                  the discord for more information.
                </div>
              )}

              {!authenticatedUser && n_users != null && <LoginForm />}

              {authenticatedUser && n_users != null && (
                <div>
                  <div className="text-center">
                    <div>
                      Welcome{' '}
                      <span className="fw-bold">
                        {authenticatedUser.username}
                        <IoMdLogOut
                          title="Logout"
                          onClick={async () => {
                            const success = await logout();
                            if (success) navigate('/');
                          }}
                          className="logout mx-2"
                          style={{ cursor: 'pointer' }}
                        />
                        <Tooltip anchorSelect=".logout" place="top">
                          Log out
                        </Tooltip>
                      </span>
                    </div>
                    <div className="justify-content-center">
                      <Link
                        to="/projects"
                        className="btn btn-lg text-white fw-bold shadow-sm px-4 py-2"
                        style={{
                          background: 'linear-gradient(90deg, #ff9a3c, #ff6f3c, #ffb347)',
                          border: 'none',
                          borderRadius: '2rem',
                          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                        }}
                        onMouseOver={(e) => {
                          e.currentTarget.style.transform = 'scale(1.05)';
                          e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.2)';
                        }}
                        onMouseOut={(e) => {
                          e.currentTarget.style.transform = 'scale(1)';
                          e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                        }}
                      >
                        Go to your projects
                      </Link>
                    </div>
                    {(messages || []).length > 0 && (
                      <div style={{ maxWidth: '600px', margin: '1.5rem auto' }}>
                        {(messages || []).map((msg) => (
                          <MessageCard key={msg.id} content={msg.content} time={msg.time} />
                        ))}
                      </div>
                    )}
                    <div className="mt-3">
                      <Link
                        to="/toolbox"
                        className="text-muted"
                        style={{ fontSize: '0.85rem' }}
                        title="Split texts into smaller units to build a dataset"
                      >
                        <FaCut className="me-1" />
                        Prepare a dataset
                      </Link>
                    </div>
                  </div>
                </div>
              )}
            </center>
            <center>
              <div
                className="text-muted mt-4"
                style={{ fontSize: '0.85rem', maxWidth: '600px', margin: '2rem auto' }}
              >
                Active Tigger is an open source, community-driven tool.
                <br />
                <a
                  href="https://github.com/activetigger/activetigger/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  If you encounter a problem, reach out or open an issue.
                </a>
              </div>
            </center>
            <div style={{ height: '50px' }}></div>
          </div>
        </div>
        <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
          <div className="container text-center">
            <div
              className="ml-2 d-flex justify-content-center align-items-center"
              style={{ fontSize: '0.8rem' }}
            >
              {n_users} active users • API {version} • Client {__BUILD_DATE__}
              <a
                href="https://www.css.cnrs.fr/active-tigger/"
                target="_blank"
                rel="noopener noreferrer"
                style={{ all: 'unset', cursor: 'pointer' }}
              >
                {'\u00A0•\u00A0'}
                CREST / CSS @ IPP © 2026
              </a>
              <a
                href="https://github.com/activetigger/activetigger"
                target="_blank"
                rel="noopener noreferrer"
                className="d-flex align-items-center"
                style={{ all: 'unset', cursor: 'pointer' }}
              >
                <FaGithub className="mx-2" />
              </a>
              {'\u00A0•\u00A0'}
              <div
                className="d-flex align-items-center gap-1"
                style={{ cursor: 'pointer', opacity: developmentMode ? 1 : 0.4 }}
                onClick={toggleDevelopmentMode}
                title={developmentMode ? 'Disable experimental mode' : 'Enable experimental mode'}
              >
                <span style={{ fontSize: '1rem' }}>🧪</span>
                <span style={{ fontSize: '0.7rem' }}>experimental</span>
              </div>
            </div>
          </div>
        </footer>
      </main>
      <Notifications />
    </>
  );
};
