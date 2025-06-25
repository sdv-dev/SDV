"""Script to generate release notes."""

import argparse
from collections import defaultdict

from scripts.github_client import GithubClient

LABEL_TO_HEADER = {
    'feature request': 'New Features',
    'bug': 'Bugs Fixed',
    'internal': 'Internal',
    'maintenance': 'Maintenance',
    'customer success': 'Customer Success',
    'documentation': 'Documentation',
    'misc': 'Miscellaneous',
}
ISSUE_LABELS = [
    'documentation',
    'maintenance',
    'internal',
    'bug',
    'feature request',
    'customer success',
]
ISSUE_LABELS_ORDERED_BY_IMPORTANCE = [
    'feature request',
    'customer success',
    'bug',
    'documentation',
    'internal',
    'maintenance',
]
NEW_LINE = '\n'


def _get_milestone_number(client, milestone_title):
    query_params = {'milestone': milestone_title, 'state': 'all', 'per_page': 100}
    response = client.get(
        github_org='sdv-dev', repo='sdv', endpoint='milestones', query_params=query_params
    )
    body = response.json()
    if response.status_code != 200:
        raise Exception(str(body))

    milestones = body
    for milestone in milestones:
        if milestone.get('title') == milestone_title:
            return milestone.get('number')

    raise ValueError(f'Milestone {milestone_title} not found in past 100 milestones.')


def _get_issues_by_milestone(milestone):
    # get milestone number
    client = GithubClient()
    milestone_number = _get_milestone_number(client, milestone)
    page = 1
    query_params = {'milestone': milestone_number, 'state': 'all'}
    issues = []
    while True:
        query_params['page'] = page
        response = client.get(
            github_org='sdv-dev',
            repo='sdv',
            endpoint='issues',
            query_params=query_params,
            timeout=10,
        )
        body = response.json()
        if response.status_code != 200:
            raise Exception(str(body))

        issues_on_page = body
        if not issues_on_page:
            break

        # Filter our PRs
        issues_on_page = [issue for issue in issues_on_page if issue.get('pull_request') is None]
        issues.extend(issues_on_page)
        page += 1

    return issues


def _get_issues_by_category(release_issues):
    category_to_issues = defaultdict(list)

    for issue in release_issues:
        issue_title = issue['title']
        issue_number = issue['number']
        issue_url = issue['html_url']
        line = f'* {issue_title} - Issue [#{issue_number}]({issue_url})'
        assignee = issue.get('assignee')
        if assignee:
            login = assignee['login']
            line += f' by @{login}'

        # Check if any known label is marked on the issue
        labels = [label['name'] for label in issue['labels']]
        found_category = False
        for category in ISSUE_LABELS:
            if category in labels:
                category_to_issues[category].append(line)
                found_category = True
                break

        if not found_category:
            category_to_issues['misc'].append(line)

    return category_to_issues


def _create_release_notes(issues_by_category, version, date):
    title = f'## v{version} - {date}'
    release_notes = f'{title}{NEW_LINE}{NEW_LINE}'

    for category in ISSUE_LABELS_ORDERED_BY_IMPORTANCE + ['misc']:
        issues = issues_by_category.get(category)
        if issues:
            section_text = (
                f'### {LABEL_TO_HEADER[category]}{NEW_LINE}{NEW_LINE}'
                f'{NEW_LINE.join(issues)}{NEW_LINE}{NEW_LINE}'
            )

            release_notes += section_text

    return release_notes


def update_release_notes(release_notes):
    """Add the release notes for the new release to the ``HISTORY.md``."""
    file_path = 'HISTORY.md'
    with open(file_path, 'r') as history_file:
        history = history_file.read()

    token = '# Release Notes\n\n'
    split_index = history.find(token) + len(token)
    header = history[:split_index]
    new_notes = f'{header}{release_notes}{history[split_index:]}'

    with open(file_path, 'w') as new_history_file:
        new_history_file.write(new_notes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, help='Release version number (ie. v1.0.1)')
    parser.add_argument('-d', '--date', type=str, help='Date of release in format YYYY-MM-DD')
    args = parser.parse_args()
    release_number = args.version
    release_issues = _get_issues_by_milestone(release_number)
    issues_by_category = _get_issues_by_category(release_issues)
    release_notes = _create_release_notes(issues_by_category, release_number, args.date)
    update_release_notes(release_notes)
