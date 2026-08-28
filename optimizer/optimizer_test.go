package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

type TestModel struct {
	P layer.Parameter
}

func (m *TestModel) Params() layer.Parameters {
	return map[string]layer.Parameter{
		"p": m.P,
	}
}

func ExampleParams() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	params := optimizer.Params(m, nil)
	for _, p := range params {
		fmt.Println(p)
	}

	// Output:
}
