# EOS Skill Reporting Contract

When running the eos-models skill, always produce ALL of the following:

1. Terminal summary: MSE table ranking all models with fitted coefficients
2. Brief interpretation (3-6 bullets):
   - Coefficient meaning: E₀/P₀ baseline, V₀ equilibrium volume, B₀/K₀ bulk modulus, Bₚ/Kp pressure derivative
   - Fit quality: lower MSE = better agreement
   - Stiffness context: larger B₀/K₀ = stiffer material
   - Unit hint for bulk modulus
   - Plausibility check: flag non-physical values (B₀ ≤ 0, extreme Bₚ, unrealistic V₀)
3. Generated file paths

Do not print results and stop. Always include the table and interpretation together.

