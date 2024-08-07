import cx from 'classnames';
import { FC, useState } from 'react';
import { IoMdLogIn, IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';

import { useAuth } from '../../core/auth';

const PAGES: { id: string; label: string; href: string }[] = [
  { id: 'login', label: 'Login', href: '/login' },
  { id: 'projects', label: 'Projects', href: '/projects' },
  { id: 'help', label: 'Help', href: '/help' },
];

interface NavBarPropsType {
  currentPage?: string;
  projectName?: string;
}

const NavBar: FC<NavBarPropsType> = ({ currentPage, projectName }) => {
  const { authenticatedUser, logout } = useAuth();
  const navigate = useNavigate();

  const [expanded, setExpanded] = useState<boolean>(false);

  return (
    <div className="bg-primary">
      <nav className="navbar navbar-dark navbar-expand-lg bg-primary">
        <div className="container">
          <Link className="navbar-brand" to="/">
            Active Tigger
          </Link>
          <button
            className="navbar-toggler"
            type="button"
            aria-controls="navbarSupportedContent"
            aria-expanded={expanded}
            aria-label="Toggle navigation"
            onClick={() => setExpanded((e) => !e)}
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div
            className={cx('navbar-collapse', expanded ? 'expanded' : 'collapse')}
            id="navbarSupportedContent"
          >
            <ul className="navbar-nav me-auto mb-2 mb-lg-0">
              {PAGES.map(({ id, label, href }) => (
                <li key={id} className="nav-item">
                  <Link
                    className={cx('nav-link', currentPage === id && 'active')}
                    aria-current={currentPage === id ? 'page' : undefined}
                    to={href}
                  >
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
            {/* TODO: add login and logout action items here */}
            {authenticatedUser ? (
              <span className="navbar-text navbar-text-margins">
                Logged as {authenticatedUser.username} -{' '}
                <button
                  className="btn btn-primary"
                  onClick={async () => {
                    const success = await logout();
                    if (success) navigate('/');
                  }}
                >
                  {' '}
                  <IoMdLogOut title="Logout" />
                </button>
              </span>
            ) : (
              <Link to="/login">
                <IoMdLogIn title="login" />
              </Link>
            )}
            {projectName && (
              <span className="navbar-text navbar-text-margins">Project {projectName}</span>
            )}
          </div>
        </div>
      </nav>
    </div>
  );
};

export default NavBar;
