using Waveq
using LinearAlgebra: norm

# ------------------------------------------------------------------
# Test iczt_ft2 round-trip accuracy
# ------------------------------------------------------------------

function test_roundtrip_integer_ratio()
    println("=== Test 1: Round-trip with integer-ratio spectrum ===")
    nx, ny = 63, 47
    x_ = collect(range(-1.0, 1.0; length=nx))
    y_ = collect(range(-0.75, 0.75; length=ny))
    f = rand(ComplexF64, ny, nx)

    nxi = 3 * nx   # 189, integer ratio
    neta = 3 * ny  # 141, integer ratio
    U, xi, eta = czt_ft2(f, x_, y_, nxi, neta)
    f_rec = iczt_ft2(U, xi, eta, x_, y_)

    err = norm(f_rec - f) / norm(f)
    println("Round-trip error (integer ratio): ", err)
    @assert err < 1e-10 "Round-trip failed for integer ratio"
    println("PASS\n")
end

function test_roundtrip_noninteger_ratio()
    println("=== Test 2: Round-trip with non-integer-ratio spectrum ===")
    nx, ny = 65, 49
    x_ = collect(range(-1.0, 1.0; length=nx))
    y_ = collect(range(-0.75, 0.75; length=ny))
    f = rand(ComplexF64, ny, nx)

    nxi = 65
    neta = 49
    U, xi, eta = czt_ft2(f, x_, y_, nxi, neta)
    f_rec = iczt_ft2(U, xi, eta, x_, y_)

    err = norm(f_rec - f) / norm(f)
    println("Round-trip error (non-integer ratio): ", err)
    @assert err < 0.2 "Round-trip failed for non-integer ratio"
    println("PASS\n")
end

function test_dummy_dimensions()
    println("=== Test 3: Dummy dimensions ===")
    
    # Dummy x (integer ratio in y)
    nx, ny = 1, 47
    x_ = [0.0]
    y_ = collect(range(-0.75, 0.75; length=ny))
    f = rand(ComplexF64, ny, nx)
    U, xi, eta = czt_ft2(f, x_, y_, 1, ny)
    f_rec = iczt_ft2(U, xi, eta, x_, y_)
    err = norm(f_rec - f) / norm(f)
    println("Dummy x round-trip error: ", err)
    @assert err < 1e-10 "Dummy x round-trip failed"

    # Dummy y (integer ratio in x)
    nx, ny = 63, 1
    x_ = collect(range(-1.0, 1.0; length=nx))
    y_ = [0.0]
    f = rand(ComplexF64, ny, nx)
    U, xi, eta = czt_ft2(f, x_, y_, nx, 1)
    f_rec = iczt_ft2(U, xi, eta, x_, y_)
    err = norm(f_rec - f) / norm(f)
    println("Dummy y round-trip error: ", err)
    @assert err < 1e-10 "Dummy y round-trip failed"

    println("PASS\n")
end

test_roundtrip_integer_ratio()
test_roundtrip_noninteger_ratio()
test_dummy_dimensions()
println("All iczt_ft2 tests passed!")
