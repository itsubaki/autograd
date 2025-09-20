package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func Sigmoid(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SigmoidT{}}).First(x...)
}

type SigmoidT struct {
	y *variable.Variable
}

func (f *SigmoidT) Forward(x ...*variable.Variable) []*variable.Variable {
	tanh := tensor.Tanh(tensor.MulC(0.5, x[0].Data)) // tanh(0.5 * x)
	y := tensor.AddC(0.5, tensor.MulC(0.5, tanh))    // 0.5 + 0.5 * tanh(0.5 * x)

	f.y = variable.NewFrom(y)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SigmoidT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		Mul(gy[0], Mul(f.y, SubC(1.0, f.y))), // gy * y * (1 - y)
	}
}
