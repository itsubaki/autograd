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

	y := vector.F(f.x, square)
	return []variable.Data{y}
}

func (f *SquareT) Backward(gy ...variable.Data) []variable.Data {
	return []variable.Data{vector.F2(f.x, gy[0], dsquare)}
}

func square(a float64) float64 { return a * a }

func dsquare(a, b float64) float64 { return 2 * a * b }
