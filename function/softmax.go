package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SoftmaxT{}}).ApplyAndFirst(x...)
}

type SoftmaxT struct {
	y *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max := matrix.BroadcastTo(matrix.Shape(x[0].Data), matrix.MaxAxis1(x[0].Data)) // max(x, axis=1)
	y := matrix.Exp(matrix.Sub(x[0].Data, max))                                    // exp(x - max)
	sum := matrix.BroadcastTo(matrix.Shape(y), matrix.SumAxis1(y))                 // sum(y, axis=1)
	y = matrix.Div(y, sum)                                                         // y / sum

	f.y = variable.NewOf(y...)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx := Mul(gy[0], f.y)               // gx = gy * y
	sumgx := SumTo(len(gx.Data), 1)(gx) // sumgx = sum(gx, axis=1)
	gx = Sub(gx, Mul(f.y, sumgx))       // gx = gx - y * sumgx

	return []*variable.Variable{
		gx,
	}
}
