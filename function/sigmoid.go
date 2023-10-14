package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Sigmoid(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *variable.Variable
}

func (f *SigmoidT) Forward(x ...*variable.Variable) []*variable.Variable {
	tanh := matrix.Tanh(matrix.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := matrix.AddC(0.5, matrix.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = variable.NewOf(y...)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(gy[0], Mul(f.y, SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}
