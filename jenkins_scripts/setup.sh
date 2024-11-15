set -eu -o pipefail # fail on error and report it, debug all lines

sudo -n true
test $? -eq 0 || exit 1 "you should have sudo privilege to run this script"

# this will check if the host has the necessary packages to start,
git --version
pip -V

# Check if Java is installed
if command -v java &>/dev/null; then
    # Java is installed, set JAVA_HOME
    export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
    echo "JAVA_HOME is set to: $JAVA_HOME"
else
    echo "Java is not installed. Please install Java and try again."
fi

export JAVA_HOME
groovy -version