package vector

func NewLike(v []float64) []float64 {
	return make([]float64, len(v))
}

func OneLike(v []float64) []float64 {
	return F(make([]float64, len(v)), func(a float64) float64 { return a + 1 })
}

func AddC(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return a + c })
}

func SubC(v []float64, c float64) []float64 {
	return F(v, func(a float64) float64 { return a - c })
}

func F(v []float64, f func(a float64) float64) []float64 {
	out := NewLike(v)
	for i := range v {
		out[i] = f(v[i])
	}

	return out
}

func F2(v, w []float64, f func(a, b float64) float64) []float64 {
	out := NewLike(v)
	for i := range v {
		out[i] = f(v[i], w[i])
	}

	return out
}
