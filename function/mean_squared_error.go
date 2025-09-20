package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func MeanSquaredError(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MeanSquaredErrorT{}}).First(x...)
}

type MeanSquaredErrorT struct {
	x0, x1 *variable.Variable
}

func (f *MeanSquaredErrorT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x0, f.x1 = x[0], x[1]

	diff := tensor.Sub(x[0].Data, x[1].Data)
	N := diff.Shape[0]
	y := tensor.Sum(tensor.Mul(diff, diff)).At() / float64(N) // (x0 - x1)^2 / N
	return []*variable.Variable{
		variable.New(y),
	}
}

func (f *MeanSquaredErrorT) Backward(gy ...*variable.Variable) []*variable.Variable {
	diff := Sub(f.x0, f.x1)
	N := diff.Shape()[0]
	gx0 := MulC(2.0/float64(N), Mul(gy[0], diff)) // gy * (x0 - x1) * 2/N
	gx1 := Neg(gx0)                               // -gx0

	return []*variable.Variable{
		gx0,
		gx1,
	}
}
