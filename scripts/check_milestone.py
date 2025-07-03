"""Script for checking milestones are attached to issues linked to PRs."""

import argparse

from scripts.github_client import GithubClient, GithubGraphQLClient

CLOSING_ISSUES_QUERY = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
        repository(owner: $owner, name: $repo) {
            pullRequest(number: $prNumber) {
                closingIssuesReferences(first: 10) {
                    nodes {
                        number
                    }
                }
            }
        }
    }
"""
OWNER = 'sdv-dev'
REPO = 'sdv'


def _get_linked_issues(
    github_client: GithubClient, graph_client: GithubGraphQLClient, pr_number: int
):
    pr_number = pr_number
    results = graph_client.query(
        query=CLOSING_ISSUES_QUERY, owner=OWNER, repo=REPO, prNumber=pr_number
    )
    pr = results.json().get('data', {}).get('repository', {}).get('pullRequest', {})
    issues = pr.get('closingIssuesReferences', {}).get('nodes', [])
    issue_numbers = [issue['number'] for issue in issues]
    linked_issues = []
    for number in issue_numbers:
        issue = github_client.get(github_org=OWNER, repo=REPO, endpoint=f'issues/{number}')
        linked_issues.append(issue.json())

    return linked_issues


def _post_comment(github_client: GithubClient, pr_number: int):
    comment = (
        'This Pull Request is not linked to an issue. To ensure our community is able to '
        'accurately track resolved issues, please link any issue that will be closed by this PR!'
    )
    github_client.post(
        github_org=OWNER,
        repo=REPO,
        endpoint=f'issues/{pr_number}/comments',
        payload={'body': comment},
    )


def check_for_milestone(pr_number: int):
    """Check that the pull request is linked to an issue and that the issue has a milestone.

    Args:
        pr_number (int): The string representation of the Pull Request number.
    """
    github_client = GithubClient()
    graphql_client = GithubGraphQLClient()
    linked_issues = _get_linked_issues(github_client, graphql_client, pr_number)
    if not linked_issues:
        _post_comment(github_client, pr_number)

    milestones = set()
    for issue in linked_issues:
        milestone = issue.get('milestone')
        if not milestone:
            raise Exception(f'No milestone attached to issue number {issue.get("number")}')

        milestones.add(milestone.get('title'))

    if len(milestones) > 1:
        raise Exception('This PR resolves issues with multiple different milestones.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pr-number', type=int, help='The number of the pull request')
    args = parser.parse_args()
    check_for_milestone(args.pr_number)
