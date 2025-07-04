"""
Main entry point for AgentChef CLI.
"""

import click
from agentChef.cli.help_texts import MAIN_HELP, ARGS_VERBOSE_HELP, ARGS_CONFIG_HELP
from agentChef.cli.cmd.research_cmd import research
from agentChef.cli.cmd.file_cmd import files
from agentChef.cli.cmd.build_cmd import build

@click.group(help=MAIN_HELP)
@click.version_option(message='%(prog)s %(version)s')
@click.option('--verbose', is_flag=True, help=ARGS_VERBOSE_HELP)
@click.option('--config', help=ARGS_CONFIG_HELP)
def cli(verbose, config):
    """AgentChef CLI - AI Agent Development Framework."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

# Add command groups
cli.add_command(research)
cli.add_command(files)
cli.add_command(build)

if __name__ == "__main__":
    cli()