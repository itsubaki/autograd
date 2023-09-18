package numerical_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/numerical"
	"github.com/itsubaki/autograd/variable"
)

func ExampleDiff() {
	// p22
	v := []*variable.Variable{variable.New(2.0)}
	f := F.Square

	fmt.Println(numerical.Diff(f, v))

	// Output:
	// variable[4.000000000004]
}
