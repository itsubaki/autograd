package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Softmax(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &SoftmaxT{}}).First(x...)
}

type SoftmaxT struct {
	y *variable.Variable
}

func (f *SoftmaxT) Forward(x ...*variable.Variable) []*variable.Variable {
	max := matrix.MaxAxis1(x[0].Data)              // max(x, axis=1)
	expy := matrix.Exp(matrix.Sub(x[0].Data, max)) // expy = exp(x - max)
	sumy := matrix.SumAxis1(expy)                  // sumy = sum(expy, axis=1)
	y := matrix.Div(expy, sumy)                    // y = expy / sumy

	f.y = variable.From(y)
	return []*variable.Variable{
		f.y,
	}
}

func (f *SoftmaxT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gyy := Mul(gy[0], f.y) // gyy = gy * y
	N := gyy.Shape()[0]
	sum := SumTo(N, 1)(gyy) // sum = sum(gx, axis=1)

	return []*variable.Variable{
		Sub(gyy, Mul(f.y, sum)), // gyy - y * sum
	}
}
