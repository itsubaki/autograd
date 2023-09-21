package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleOpts() {
	fmt.Printf("%#v", variable.Opts{Retain: true})

	// Output:
	// variable.Opts{Retain:true}
}
