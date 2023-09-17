package function

import (
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Square(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SquareT{}}).Apply(x...)[0]
}

type SquareT struct {
	x variable.Data
}

func (f *SquareT) Forward(x ...variable.Data) []variable.Data {
	f.x = x[0]

	square := func(a float64) float64 { return a * a }
	y := vector.F(f.x, square)
	return []variable.Data{y}
}

func (f *SquareT) Backward(gy ...variable.Data) []variable.Data {
	dsquare := func(a, b float64) float64 { return 2 * a * b }
	grad := vector.F2(f.x, gy[0], dsquare)
	return []variable.Data{grad}
}
