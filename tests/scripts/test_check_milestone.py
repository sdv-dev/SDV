from unittest.mock import Mock, patch

import pytest
from scripts.check_milestone import CLOSING_ISSUES_QUERY, check_for_milestone


@patch('scripts.check_milestone.GithubGraphQLClient')
@patch('scripts.check_milestone.GithubClient')
def test_check_for_milestone(mock_github_client, mock_github_graph_ql_client):
    # Setup
    mock_query_result = Mock()
    mock_query_result.json.return_value = {
        'data': {
            'repository': {
                'pullRequest': {'closingIssuesReferences': {'nodes': [{'number': 1234}]}}
            }
        }
    }
    mock_get_result = Mock()
    mock_get_result.json.return_value = {'milestone': '1.1.1'}
    mock_github_graph_ql_client.return_value.query.return_value = mock_query_result
    mock_github_client.return_value.get.return_value = mock_get_result

    # Run
    check_for_milestone(1222)

    # Assert
    mock_github_graph_ql_client.return_value.query.assert_called_with(
        query=CLOSING_ISSUES_QUERY, owner='sdv-dev', repo='sdv', prNumber=1222
    )
    mock_github_client.return_value.get.assert_called_once_with(
        github_org='sdv-dev', repo='sdv', endpoint='issues/1234'
    )
    mock_github_client.return_value.post.assert_not_called()


@patch('scripts.check_milestone.GithubGraphQLClient')
@patch('scripts.check_milestone.GithubClient')
def test_check_for_milestone_no_milestone(mock_github_client, mock_github_graph_ql_client):
    # Setup
    mock_query_result = Mock()
    mock_query_result.json.return_value = {
        'data': {
            'repository': {
                'pullRequest': {'closingIssuesReferences': {'nodes': [{'number': 1234}]}}
            }
        }
    }
    mock_get_result = Mock()
    mock_get_result.json.return_value = {'number': 1234}
    mock_github_graph_ql_client.return_value.query.return_value = mock_query_result
    mock_github_client.return_value.get.return_value = mock_get_result

    # Run and assert
    expected_message = 'No milestone attached to issue number 1234'
    with pytest.raises(Exception, match=expected_message):
        check_for_milestone(1222)


@patch('scripts.check_milestone.GithubGraphQLClient')
@patch('scripts.check_milestone.GithubClient')
def test_check_for_milestone_no_linked_issues(mock_github_client, mock_github_graph_ql_client):
    # Setup
    mock_query_result = Mock()
    mock_query_result.json.return_value = {
        'data': {'repository': {'pullRequest': {'closingIssuesReferences': {'nodes': []}}}}
    }
    mock_github_graph_ql_client.return_value.query.return_value = mock_query_result

    # Run
    check_for_milestone(1222)

    # Assert
    mock_github_graph_ql_client.return_value.query.assert_called_once_with(
        query=CLOSING_ISSUES_QUERY, owner='sdv-dev', repo='sdv', prNumber=1222
    )
    mock_github_client.return_value.post.assert_called_once_with(
        github_org='sdv-dev',
        repo='sdv',
        endpoint='issues/1222/comments',
        payload={
            'body': 'This Pull Request is not linked to an issue. To ensure our community is able '
            'to accurately track resolved issues, please link any issue that will be closed by '
            'this PR!'
        },
    )
