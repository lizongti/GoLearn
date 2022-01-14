package display

import "fmt"

func Line() {
	fmt.Println("")
}

func Reset() {
	count = 0
}

var count int = 0

func Section(s string, v interface{}) {
	count++
	fmt.Printf("\n----------------------------[%d.%s]----------------------------\n%v\n", count, s, v)
}
