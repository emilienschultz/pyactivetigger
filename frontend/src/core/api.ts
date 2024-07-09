import createClient from 'openapi-fetch';
import { useCallback } from 'react';

import type { components, paths } from '../generated/openapi';
import { ProjectDataModel } from '../types';
import config from './config';
import { useAppContext } from './context';

const api = createClient<paths>({ baseUrl: `${config.api.url}` });

export type LoginParams = components['schemas']['Body_login_for_access_token_token_post'];

export async function login(params: LoginParams) {
  const res = await api.POST('/token', {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params,
    bodySerializer: (body) => new URLSearchParams(body as Record<string, string>),
  });
  if (res.data && !res.error) return res.data;
  //else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
  else throw new Error("Try again");


}

export async function me(token: string) {
  const res = await api.GET('/users/me', {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (res.data) return res.data;
  //else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
}

export async function userProjects(username: string) {
  // API /session should be rewritten at some point
  const res = await api.GET('/projects', {
    params: {
      header: { username },
    },
  });

  if (res.data && !res.error) return res.data.projects;
  else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
}

export function useCreateProject() {
  const { appContext } = useAppContext();
  const createProject = useCallback(async (project: ProjectDataModel) => {
    if (appContext.user) {
      const res = await api.POST('/projects/new', {
        headers: {
          Authorization: `Bearer ${appContext.user.access_token}`,
          username: appContext.user.username,
        },
        params: { header: { username: appContext.user.username } },
        body: project,
      });
      if (res.error)
        throw new Error(
          res.error.detail ? res.error.detail?.map((d) => d.msg).join('; ') : res.error.toString(),
        );
    }
    //TODO: notify
  }, []);
  return createProject;
}

export function useGetProject() {
  const { appContext } = useAppContext();
  const getProject = useCallback(async (projectName: string) => {
    if (appContext.user) {
      const res = await api.GET('/state/{project_name}', {
        headers: {
          Authorization: `Bearer ${appContext.user.access_token}`,
          username: appContext.user.username,
        },
        params: {
          header: { username: appContext.user.username },
          path: { project_name: projectName },
        },
      });
      if (res.error)
        throw new Error(
          res.error.detail ? res.error.detail?.map((d) => d.msg).join('; ') : res.error.toString(),
        );
      return res.data.params;
    }
    //TODO: notify
  }, []);
  return getProject;
}
