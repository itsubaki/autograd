package optimizer_test

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSGD() {
	m := model.NewMLP([]int{10, 1})
	o := optimizer.SGD{
		LearningRate: 0.2,
	}

	x := variable.Rand(10, 1)
	y := variable.Rand(10, 1)

	yPred := m.Forward(x)
	loss := F.MeanSquaredError(y, yPred)

	m.Cleargrads()
	loss.Backward()
	o.Update(m)

	// Output:
}
