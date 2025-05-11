package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func LinearSimple(x, w *variable.Variable, b ...*variable.Variable) *variable.Variable {
	t := MatMul(x, w)
	if len(b) == 0 {
		return t
	}

	y := Add(t, b[0])
	t.Data = matrix.Matrix{} // erase
	return y
}
