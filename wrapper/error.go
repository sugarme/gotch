package wrapper

/*
 * import "C"
 *
 * import (
 *   "fmt"
 * )
 *
 * // ptrToString returns nil on the null pointer. If not null,
 * // the pointer gets freed.
 * // NOTE: C does not have exception design. C++ throws exception
 * // to stderr. This code to check stderr for any err message,
 * // if it exists, takes it and frees up C pointer.
 * func ptrToString(ptr *C.c_char) string {
 *   var str string
 *   if !ptr.is_null() {
 *     // TODO: implement this
 *     // str := GET_ERROR_FROM C std::err
 *     C.free(ptr)
 *     return str
 *   } else {
 *     return ""
 *   }
 * }
 *
 * // readAndCleanError wraps error handling and C memory free up
 * func UnsafeTorch(f func()) (retF func(), err error) {
 *
 *   var str string
 *   // TODO: implement this
 *   // str := ptrToString(torch_sys.get_and_reset_last_err())
 *   if str != "" {
 *     err = fmt.Errorf("Unsafe error: %v\n", err.Error())
 *     return nil, err
 *   } else {
 *     return f, nil
 *   }
 * } */
