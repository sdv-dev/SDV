"""Clients for making requests to Github APIs."""

import os

import requests


class BaseClient:
    """Base GitHub client."""

    def __init__(self):
        token = os.getenv('GH_ACCESS_TOKEN')
        self.headers = {'Authorization': f'Bearer {token}'}


class GithubGraphQLClient(BaseClient):
    """Client for GitHub GraphQL."""

    def __init__(self):
        super().__init__()
        self.base_url = 'https://api.github.com/graphql'

    def query(self, query: str, **kwargs):
        """Run a query on Github GraphQL.

        Args:
            query (str):
                The query to run.
            kwargs (dict):
                Any key-word arguments needed for the query.

        Returns:
            requests.models.Response
        """
        response = requests.post(
            self.base_url, json={'query': query, 'variables': kwargs}, headers=self.headers
        )
        return response


class GithubClient(BaseClient):
    """Client for GitHub API."""

    def __init__(self):
        super().__init__()
        self.base_url = 'https://api.github.com/repos'

    def _construct_url(self, github_org: str, repo: str, resource: str, id: str | None = None):
        url = f'{self.base_url}/{github_org}/{repo}/{resource}'
        if id:
            url += f'/{id}'
        return url

    def list(
        self,
        github_org: str,
        repo: str,
        resource: str,
        query_params: dict | None = None,
        timeout: int | None = None,
    ):
        """Get all values for a resource from a GitHub repository.

        Args:
            github_org (str):
                The name of the GitHub organization to search.
            repo (str):
                The name of the repository to search.
            resource (str):
                The name of the resource we are getting. For example, issues. This means we'd be
                making a request to https://api.github.com/repos/{github_org}/{repo}/{resource}.
            query_params (dict):
                A dictionary mapping any query parameters to the desired value. Defaults to None.
            timeout (int):
                How long to wait before the request times out. Defaults to None.

        Returns:
            requests.models.Response
        """
        url = self._construct_url(github_org, repo, resource)
        return requests.get(url, headers=self.headers, params=query_params, timeout=timeout)

    def get(
        self,
        github_org: str,
        repo: str,
        endpoint: str,
        query_params: dict | None = None,
        timeout: int | None = None,
    ):
        """Get a specific value of a resource from an endpoint in the GitHub API.

        Args:
            github_org (str):
                The name of the GitHub organization to search.
            repo (str):
                The name of the repository to search.
            endpoint (str):
                The endpoint for the resource. For example, issues. This means we'd be
                making a request to https://api.github.com/repos/{github_org}/{repo}/issues.
            query_params (dict):
                A dictionary mapping any query parameters to the desired value. Defaults to None.
            timeout (int):
                How long to wait before the request times out. Defaults to None.

        Returns:
            requests.models.Response
        """
        url = self._construct_url(github_org, repo, endpoint)
        return requests.get(url, headers=self.headers, params=query_params, timeout=timeout)

    def post(self, github_org: str, repo: str, endpoint: str, payload: dict):
        """Post to an endpooint in the GitHub API.

        Args:
            github_org (str):
                The name of the GitHub organization to search.
            repo (str):
                The name of the repository to search.
            endpoint (str):
                The endpoint for the resource. For example, issues. This means we'd be
                making a request to https://api.github.com/repos/{github_org}/{repo}/issues.
            payload (dict):
                The payload to post.

        Returns:
            requests.models.Response
        """
        url = self._construct_url(github_org, repo, endpoint)
        return requests.post(url, headers=self.headers, json=payload)
