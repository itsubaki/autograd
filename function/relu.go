package function

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func ReLU(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &ReLUT{}}).ApplyAndFirst(x...)
}

type ReLUT struct {
	x *variable.Variable
}

func (f *ReLUT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x = x[0]

	y := matrix.F(x[0].Data, max)
	return []*variable.Variable{
		variable.NewOf(y...),
	}
}

func (f *ReLUT) Backward(gy ...*variable.Variable) []*variable.Variable {
	mask := matrix.Mask(f.x.Data, relu)
	return []*variable.Variable{
		Mul(gy[0], variable.NewOf(mask...)), // gy * mask
	}
}

func max(v float64) float64 { return math.Max(v, 0.0) }

func relu(v float64) bool { return v > 0 }
