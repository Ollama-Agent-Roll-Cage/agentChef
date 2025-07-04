"""
Command-line interface for AgentChef.

This module provides the main command-line interface and subcommands
for AgentChef operations.
"""

import click

from oarc_log import enable_debug_logging

from agentChef.cli.help_texts import MAIN_HELP, ARGS_VERBOSE_HELP, ARGS_CONFIG_HELP
from agentChef.config.config import apply_config_file
from agentChef.cli.cmd import (
    research,
    files,  # Add files command
    build,
)

@click.group(help=MAIN_HELP)
@click.version_option(message='%(prog)s %(version)s')
@click.option('--verbose', is_flag=True, help=ARGS_VERBOSE_HELP, callback=enable_debug_logging)
@click.option('--config', help=ARGS_CONFIG_HELP, callback=apply_config_file)
def cli(verbose, config):
    """AgentChef CLI - AI Agent Development Framework."""
    pass

# Add commands
cli.add_command(research)
cli.add_command(files)  # Add files command
cli.add_command(build)

if __name__ == "__main__":
    cli()
