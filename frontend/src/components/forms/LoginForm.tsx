import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';

import { LoginParams, login, me } from '../../core/api';
import { useAppContext } from '../../core/context';

export const LoginForm: FC = () => {
  const { setAppContext } = useAppContext();
  const { handleSubmit, register } = useForm<LoginParams>();
  const navigate = useNavigate();
  const [error, setError] = useState(String);

  const onSubmit: SubmitHandler<LoginParams> = async (data) => {
    try {
      //catch the error throw by login ?
      const response = await login(data);

      console.log(response);
      if (response.access_token) {
        const user = await me(response.access_token);
        if (user) {
          // if user authentified
          setAppContext({ user: { ...user, access_token: response.access_token } });
          navigate('/projects'); //redirect to the projects page
        } else {
          setAppContext({ user: undefined });
          setError('Error in user authentification');
        }
      }
    } catch (e) {
      setError('Error in user authentification');
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>Connect to the service</div>
      <input className="form-control form-appearance mt-2" type="text" {...register('username')} />
      <input className="form-control mt-2" type="password" {...register('password')} />
      <button className="btn btn-primary btn-validation">Login</button>
      {error && (
        <div className="alert alert-danger mt-3" role="alert">
          {error}
        </div>
      )}
    </form>

    // TODO : rediriger vers l'application si validé ?
  );
};
