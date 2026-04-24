# Contributing to Gold Price Forecasting

Thank you for your interest in contributing! Here's how to get started.

## How to Contribute

### Reporting Issues
- Use the GitHub Issues tab
- Include a clear title, description, and steps to reproduce
- Attach relevant error messages or screenshots

### Submitting Changes

1. **Fork** the repository
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/gold-forecasting.git
   cd gold-forecasting
   ```
3. **Create a branch** for your feature or fix
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following the code style below
5. **Test** your changes by running the notebook end-to-end
6. **Commit** with a descriptive message
   ```bash
   git commit -m "feat: add SARIMA model comparison"
   ```
7. **Push** and open a Pull Request

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names (no single letters except loop indices)
- Add docstrings to all functions using NumPy style
- Keep functions focused — one function, one purpose
- Use type hints where possible

## Ideas for Contributions

- Add SARIMA or TBATS models
- Implement Bayesian forecasting (e.g. PyMC)
- Add more macro regressors (DXY, real rates, central bank reserves)
- Improve LSTM architecture (Attention, Transformer)
- Add silver/platinum price correlation analysis
- Port visualizations to Streamlit dashboard
- Add unit tests

## Questions?

Open an issue with the label `question` — happy to help!
