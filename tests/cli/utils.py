def run_version(runner, command, version):
    result = runner.invoke(command, args=["--version"], catch_exceptions=False)
    assert result.exit_code == 0
    assert version in result.output
