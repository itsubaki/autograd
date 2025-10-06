package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdamW() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	o := optimizer.AdamW{
		Adam: optimizer.Adam{
			Alpha: 0.001,
			Beta1: 0.9,
			Beta2: 0.999,
		},
		WeightDecay: 0.1,
	}

	o.Update(m)
	fmt.Println(p)

	o.Update(m)
	fmt.Println(p)

	// Output:
	// variable(0.998968377539626)
	// variable(0.9979448703665578)
}
