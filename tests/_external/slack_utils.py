"""Utility functions for Slack integration."""

import os

from slack_sdk import WebClient


def _get_slack_client():
    """Create an authenticated Slack client.

    Returns:
        WebClient:
            An authenticated Slack WebClient instance.
    """
    token = os.getenv('SLACK_TOKEN')
    client = WebClient(token=token)
    return client


def post_slack_message(channel, text):
    """Post a message to a Slack channel.

    Args:
        channel (str):
            The name of the channel to post to.
        text (str):
            The message to send to the channel.

    Returns:
        SlackResponse:
            Response from Slack API call
    """
    client = _get_slack_client()
    response = client.chat_postMessage(channel=channel, text=text)
    if not response['ok']:
        error = response.get('error', 'unknown_error')
        msg = f'{error} occured trying to post message to {channel}'
        raise RuntimeError(msg)

    return response


def post_slack_message_in_thread(channel, text, thread_ts):
    """Post a message as a threaded reply in a Slack channel.

    Args:
        channel (str):
            The name of the channel to post to.
        text (str):
            The message to send as a reply in the thread.
        thread_ts (str):
            The timestamp of the message that starts the thread.

    Returns:
        SlackResponse:
            Response from Slack API call.
    """
    client = _get_slack_client()
    response = client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
    if not response['ok']:
        error = response.get('error', 'unknown_error')
        msg = f'{error} occurred trying to post threaded message to {channel}'
        raise RuntimeError(msg)

    return response


def update_message(channel, new_text, message_ts):
    """Update an existing message in a Slack channel.

    Args:
        channel (str):
            The name of the channel where the message is located.
        message_ts (str):
            The timestamp of the message to be edited.
        new_text (str):
            The new text to update the message with.

    Returns:
        SlackResponse:
            Response from Slack API call.
    """
    client = _get_slack_client()
    response = client.chat_update(channel=channel, ts=message_ts, text=new_text)
    if not response['ok']:
        error = response.get('error', 'unknown_error')
        msg = f'{error} occurred trying to update message in {channel}'
        raise RuntimeError(msg)

    return response
