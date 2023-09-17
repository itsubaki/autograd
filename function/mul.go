package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Mul(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &MulT{}}).Apply(x...)[0]
}

type MulT struct {
	x0, x1 variable.Data
}

func (f *MulT) Forward(x ...variable.Data) []variable.Data {
	f.x0, f.x1 = x[0], x[1]

	y := vector.F2(f.x0, f.x1, mul)
	return []variable.Data{y}
}

func (f *MulT) Backward(gy ...variable.Data) []variable.Data {
	g0 := vector.F2(f.x1, gy[0], mul)
	g1 := vector.F2(f.x0, gy[0], mul)
	return []variable.Data{g0, g1}
}

func mul(a, b float64) float64 { return a * b }
