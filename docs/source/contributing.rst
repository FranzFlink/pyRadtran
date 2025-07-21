Contributing to PyRadtran
========================

We welcome contributions to PyRadtran! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone https://github.com/yourusername/pyRadtran.git
    cd pyRadtran

3. Create a virtual environment and install development dependencies::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e ".[dev,docs]"

Development Workflow
--------------------

1. Create a new branch for your feature or bugfix::

    git checkout -b feature/your-feature-name

2. Make your changes and add tests if applicable

3. Run the test suite::

    pytest

4. Check code style::

    black .
    isort .
    flake8

5. Build and test the documentation::

    cd docs
    make clean
    make html

6. Commit your changes and push to your fork::

    git add .
    git commit -m "Add your descriptive commit message"
    git push origin feature/your-feature-name

7. Create a pull request on GitHub

Code Style
----------

We follow PEP 8 for Python code style. Use the following tools:

- **black** for automatic code formatting
- **isort** for import sorting  
- **flake8** for linting

Testing
-------

- Write tests for new functionality
- Ensure all tests pass before submitting a pull request
- Aim for good test coverage of new code

Documentation
-------------

- Update documentation for new features
- Use NumPy-style docstrings
- Include examples in docstrings where helpful
- Test that documentation builds without warnings

Submitting Changes
------------------

1. Make sure your code follows the style guidelines
2. Include tests for new functionality
3. Update documentation as needed
4. Write a clear commit message describing your changes
5. Submit a pull request with a description of what your changes do

Bug Reports
-----------

When filing bug reports, please include:

- Python version
- PyRadtran version
- Operating system
- Minimal code example that reproduces the issue
- Full error traceback

Feature Requests
----------------

For feature requests, please:

- Explain the use case for the feature
- Provide examples of how the feature would be used
- Consider whether the feature fits with the project's goals

Questions
---------

For questions about using PyRadtran, please:

- Check the documentation first
- Search existing GitHub issues
- Create a new issue with the "question" label

Thank you for contributing to PyRadtran!
