package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleParams() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	params := optimizer.Params(m, nil)
	for _, p := range params {
		fmt.Println(p)
	}

	// Output:
}
