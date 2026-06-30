# Contributing to sklearn-java

## Development Workflow

1. **Issue First** — Create an issue describing the feature or bug
2. **Feature Branch** — Branch from `develop`: `feature/my-feature`
3. **Implement** — Follow the project principles
4. **Test** — Every public method needs unit tests
5. **Validate** — Compare outputs against Python sklearn (tolerance: 1e-8)
6. **Document** — Javadoc for every public class and method
7. **PR** — Open pull request to `develop`
8. **Review** — All PRs require review before merging
9. **Merge** — Squash-merge into `develop`

## Branch Strategy

- `main` — Production-ready releases
- `develop` — Integration branch
- `feature/*` — New features
- `fix/*` — Bug fixes
- `benchmark/*` — Performance benchmarks

## Coding Standards

- Java 21+
- Follow Google Java Format
- No Python-isms: no dynamic typing, no duck typing
- Prefer readability over cleverness
- All algorithms must be deterministic
- Numerical tolerance against sklearn: 1e-8

## Validation

Every algorithm must include a Python validation script:

```python
# validation/validate_linear_regression.py
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array(...)
y = np.array(...)
model = LinearRegression().fit(X, y)
print(model.coef_)
print(model.intercept_)
```

Compare outputs with `sklearn-java` benchmarks.
