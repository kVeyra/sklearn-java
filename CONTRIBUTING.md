# Contributing to sklearn-java

Thanks for your interest in contributing! sklearn-java aims to reimplement 100% of scikit-learn's public API in pure Java. Every contribution gets us closer to that goal.

## Getting Started

1. **Pick an issue** — Check [open issues](https://github.com/kVeyra/sklearn-java/issues) or the [COVERAGE_PLAN.md](COVERAGE_PLAN.md) for unimplemented algorithms.
2. **Discuss** — Comment on the issue to let others know you're working on it.
3. **Branch** — Create a feature branch from `develop`: `git checkout -b feature/my-algorithm develop`
4. **Implement** — Follow the patterns in existing code.
5. **Submit a PR** — Open a pull request to `develop`.

## Branch Strategy

- `main` — Production-ready releases
- `develop` — Integration branch (base all PRs here)
- `feature/*` — New algorithms and features
- `fix/*` — Bug fixes
- `benchmark/*` — Performance benchmarks and validation

PRs are **squash-merged** into `develop`.

## Development Workflow

### Before implementing an algorithm

1. Fetch the Python sklearn source to understand the algorithm
2. Identify the sklearn class/function you're implementing
3. Check the module structure and existing patterns

### Implementation Requirements

| Requirement | Details |
|-------------|---------|
| **API** | Match sklearn exactly: `.fit()`, `.predict()`, `.transform()`, `.score()` |
| **Javadoc** | Every public class, constructor, and method |
| **Tests** | JUnit 5 tests covering normal cases, edge cases (empty data, single class, constant features) |
| **Determinism** | All algorithms must produce identical results given the same seed |
| **Tolerance** | Numerical results must match sklearn within 1e-8 (absolute where possible) |
| **No Python-isms** | Pure Java — no dynamic typing, no duck typing, no JNI/Python interop |

### Coding Standards

- **Java 21+** — Use records, sealed interfaces, pattern matching where appropriate
- **Google Java Format** — Consistent code style
- **No external ML libraries** — Only standard library + JUnit 5 + JaCoCo
- **Prefer readability** — Clear variable names, linear control flow
- **Validation** — Use `Validation.checkMatrix()`, `Validation.checkFitted()` from `utils`

### Test Patterns

```java
@Test
void testBasicFitAndPredict() {
    Matrix X = new Matrix(new double[][]{{1, 2}, {2, 3}, {10, 11}, {11, 12}});
    Vector y = new Vector(new double[]{0, 0, 1, 1});

    MyEstimator est = new MyEstimator(params);
    est.fit(X, y);
    Vector pred = est.predict(X);

    assertEquals(4, pred.size());
    assertTrue(est.score(X, y) > 0.8);
}
```

## Submitting a PR

1. Ensure all tests pass: `./gradlew build`
2. Ensure coverage doesn't decrease: `./gradlew jacocoTestReport`
3. Write a concise PR description referencing the issue
4. Include validation notes (how results match sklearn)

### PR Checklist

- [ ] New algorithm matches sklearn API
- [ ] JUnit 5 tests added (normal + edge cases)
- [ ] Javadoc on all public members
- [ ] Algorithm is deterministic
- [ ] Validated against sklearn (tolerance: 1e-8)
- [ ] Code follows project style
- [ ] `./gradlew build` passes (checkstyle + tests)

## Questions?

Open a [Discussion](https://github.com/kVeyra/sklearn-java/discussions) or ask in the issue tracker.
