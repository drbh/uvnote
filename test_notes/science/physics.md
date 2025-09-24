# Physics Calculations

## Newton's Laws

```python
# F = ma calculation
def calculate_force(mass, acceleration):
    """Calculate force using Newton's second law"""
    force = mass * acceleration
    return force

# Example: 10 kg object accelerating at 5 m/s^2
mass = 10  # kg
acceleration = 5  # m/s^2
force = calculate_force(mass, acceleration)
print(f"Force = {force} N")
```

## Kinetic Energy

```python
def kinetic_energy(mass, velocity):
    """Calculate kinetic energy: KE = 1/2 * m * v^2"""
    ke = 0.5 * mass * velocity**2
    return ke

# Example: 5 kg object moving at 10 m/s
m = 5  # kg
v = 10  # m/s
energy = kinetic_energy(m, v)
print(f"Kinetic Energy = {energy} J")
```