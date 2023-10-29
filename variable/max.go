package variable

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
)

func Max(x ...*Variable) *Variable {
	return (&Function{Forwarder: &MaxT{}}).First(x...)
}

type MaxT struct {
	x, y *Variable
}

func (f *MaxT) Forward(x ...*Variable) []*Variable {
	f.x = x[0]

	f.y = New(matrix.Max(x[0].Data))
	return []*Variable{
		f.y,
	}
}

func (f *MaxT) Backward(gy ...*Variable) []*Variable {
	mask := NewOf(matrix.F2(f.x.Data, f.y.Data, IsClose)...)
	return []*Variable{
		Mul(gy[0], mask),
	}
}

func IsClose(a, b float64) float64 {
	atol, rtol := 1e-08, 1e-05
	if math.Abs(a-b) < atol+rtol*math.Abs(b) {
		return 1
	}

	return 0
}
