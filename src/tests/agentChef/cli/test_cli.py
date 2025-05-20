import pytest
from click.testing import CliRunner
from unittest.mock import patch

from agentChef.cli.cmd.research_cmd import research
from agentChef.cli.cmd.build_cmd import build
from agentChef.utils.const import SUCCESS

@pytest.fixture
def cli_runner():
    """Fixture to provide a Click CLI test runner."""
    return CliRunner()

def test_research_topic(cli_runner):
    """Test research topic command."""
    with patch('agentChef.cli.cmd.research_cmd.ResearchManager') as mock_manager:
        instance = mock_manager.return_value
        instance.research_topic.return_value = {"status": "success"}
        
        result = cli_runner.invoke(research, ['topic', '--topic', 'test topic'])
        assert result.exit_code == SUCCESS

def test_research_topic_with_github(cli_runner):
    """Test research with GitHub integration."""
    with patch('agentChef.cli.cmd.research_cmd.ResearchManager') as mock_manager:
        instance = mock_manager.return_value
        instance.research_topic.return_value = {"status": "success"}
        
        result = cli_runner.invoke(research, [
            'topic', 
            '--topic', 'test topic',
            '--include-github',
            '--github-repos', 'repo1', 'repo2'
        ])
        assert result.exit_code == SUCCESS

def test_build_package(cli_runner):
    """Test build package command."""
    with patch('agentChef.cli.cmd.build_cmd.BuildUtils') as mock_utils:
        mock_utils.build_package.return_value = True
        
        result = cli_runner.invoke(build, ['package'])
        assert result.exit_code == SUCCESS

def test_build_package_with_clean(cli_runner):
    """Test build package with clean flag."""
    with patch('agentChef.cli.cmd.build_cmd.BuildUtils') as mock_utils:
        mock_utils.clean_build_directories.return_value = True
        mock_utils.build_package.return_value = True
        
        result = cli_runner.invoke(build, ['package', '--clean'])
        assert result.exit_code == SUCCESS
