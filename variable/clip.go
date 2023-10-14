package variable

import "github.com/itsubaki/autograd/matrix"

func Clip(min, max float64) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &ClipT{Min: min, Max: max}}).First
}

type ClipT struct {
	Min, Max float64
	x        *Variable
}

func (f *ClipT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	y := matrix.Clip(x[0].Data, f.Min, f.Max)
	return []*Variable{
		NewOf(y...),
	}
}

func (f *ClipT) Backward(gy ...*Variable) []*Variable {
	mask := matrix.Mask(f.x.Data, clip(f.Min, f.Max))
	return []*Variable{
		Mul(gy[0], NewOf(mask...)), // gy * mask
	}
}

func clip(min, max float64) func(v float64) bool {
	return func(v float64) bool {
		return min <= v && v <= max
	}
}
