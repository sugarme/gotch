(* Automatically generated C++ -> C -> Go bindings.
   Input: Declarations-VERSION.yaml artifact generated when building Pytorch from source.
   Run with: dune exec gen/gen.exe
*)
open Base
open Stdio

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "clone"
    ; "copy_"
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "set_data"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_amp_foreach_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args"
    ; "_validate_sparse_csr_tensor_args"
    ; "_backward"
    ; "size"
    ; "stride"
    ; "histogram_out"
    ; "histogram"
    ; "_assert_async"
    ; "gradient"
    ; "linalg_vector_norm"
    ; "linalg_vector_norm_out" 
    ; "linalg_matrix_norm"
    ; "linalg_matrix_norm_out"
    ; "_histogramdd_bin_edges"
    ; "_histogramdd_bin_edges"
    ; "_histogramdd_from_bin_cts"
    ; "_linalg_check_errors"
    ; "_index_copy_"
    ; "_native_multi_head_self_attention"
    ; "normal_out"
    ; "bernoulli_out"
    ; "nested_tensor"
    ; "arange_out"
    (* TODO: Fix `write cpp` so that it can seperate a non-return func *)
    ; "_assert_tensor_metadata"
    ; "_histogramdd_bin_edges_out"
    ; "_validate_compressed_sparse_indices"
    ; "_validate_sparse_bsc_tensor_args"
    ; "_validate_sparse_bsr_tensor_args"
    ; "_validate_sparse_compressed_tensor_args"
    ; "_validate_sparse_csc_tensor_args"
    ; "dequantize_out"
    ; "lstm_mps_backward_out"
    ; "quantize_per_tensor_out"
    ; "split_copy_out"
    ; "split_with_sizes_copy_out"
    ; "unbind_copy_out"
    ; "unsafe_split_out"
    ; "unsafe_split_with_sizes_out"
    ; "_histogramdd_from_bin_cts"
    ; "sym_numel"
    ; "sym_size"
    ; "sym_stride"
    ; "sym_storage_offset"
    ; "_sparse_semi_structured_addmm"
    ]

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like" ]

(* By default, scalar argument that have a default value are not available on
   the Rust side, this is to preserve the Rust api simplicity assuming that
   these scalars arguments are not often overriden.
   Adding function name [foo] in [with_optional_scalar_args] results in having
   explicit scalar arguments even if a default is present. *)
let with_optional_scalar_args = Set.of_list (module String) [ "arange"; "baddbmm" ]


(* 
 * let prefixed_functions =
 *   Set.of_list
 *     (module String)
 *     ["add"; "add_"; "div"; "div_"; "mul"; "mul_"; "sub"; "sub_"; "nll_loss"]
 *  *)
let excluded_prefixes = 
  ["_thnn_"
  ; "_th_"
  ; "thnn_"
  ; "th_"
  ; "_foreach"
  ; "_amp_foreach"
  ; "_nested_tensor"
  ; "_fused_adam"
  ]

let excluded_suffixes = ["_forward"; "_forward_out"]

let yaml_error yaml ~msg =
  Printf.failwithf "%s, %s" msg (Yaml.to_string_exn yaml) ()

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Int64Option
    | Double
    | DoubleOption
    | Tensor
    | TensorOption (* Tensor.t option *)
    | IntList
    | IntListOption
    | DoubleList
    | TensorOptList
    | TensorList
    | TensorOptions (* Tensor kind and device *)
    | Scalar
    | ScalarType
    | ScalarTypeOption
    | Device
    | String
    | Layout
    | LayoutOption

  type arg =
    { arg_name: string
      ; arg_type: arg_type
      ; default_value: string option
    }

  (* `Func` type   *)
  type t =
    { name: string
    ; operator_name: string
    ; overload_name: string
    ; args: arg list (* ; returns: [`fixed of int | `dynamic] *)
    ; returns: [`fixed of int | `dynamic | `bool | `int64_t | `double | `nothing]
    ; (* number of tensors that are returned *)
      kind: [`function_ | `method_] 
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some (if is_nullable then Int64Option else Int64)
    | "double" -> Some (if is_nullable then DoubleOption else Double)
    | "at::tensor" -> Some (if is_nullable then TensorOption else Tensor)
    | "at::tensoroptions" -> Some TensorOptions
    | "at::intarrayref" -> Some (if is_nullable then IntListOption else IntList)
    | "at::arrayref<double>" -> Some DoubleList
    | "const c10::list<c10::optional<at::tensor>> &" -> Some TensorOptList
    | "const at::itensorlistref &" | "at::tensorlist" -> Some TensorList
    | "at::device" -> Some Device
    | "const at::scalar &" | "at::scalar" -> Some Scalar
    | "at::scalartype" -> if is_nullable then Some ScalarTypeOption else Some ScalarType
    | "c10::string_view" -> Some String
    | "at::layout" -> Some (if is_nullable then LayoutOption else Layout)
    | _ -> None


  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | IntList | IntListOption ->
        Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
      | DoubleList -> Printf.sprintf "double *%s_data, int %s_len" arg_name arg_name
      | TensorOptList | TensorList ->
        Printf.sprintf "tensor *%s_data, int %s_len" arg_name arg_name
      | TensorOptions -> Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
      | String -> Printf.sprintf "char* %s_ptr, int %s_len" arg_name arg_name
      | Int64Option -> Printf.sprintf "int64_t %s_v, uint8_t %s_null" arg_name arg_name
      | DoubleOption -> Printf.sprintf "double %s_v, uint8_t %s_null" arg_name arg_name
      | otherwise ->
        let simple_type_cstring =
          match otherwise with
          | Bool -> "int"
          | Int64 -> "int64_t"
          | Double -> "double"
          | Tensor -> "tensor"
          | TensorOption -> "tensor"
          | ScalarType -> "int"
          | ScalarTypeOption -> "int"
          | Device -> "int"
          | Scalar -> "scalar"
          | Layout | LayoutOption -> "int8_t"
          | Int64Option
          | DoubleOption
          | String
          | IntList
          | IntListOption
          | DoubleList
          | TensorOptList
          | TensorList
          | TensorOptions -> assert false
        in
        Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | Scalar | Tensor -> "*" ^ arg_name
      | Layout -> Printf.sprintf "static_cast<at::Layout>(%s)" arg_name
      | LayoutOption ->
        Printf.sprintf
          "(%s == -1 ? c10::nullopt : \
           c10::optional<at::Layout>(static_cast<at::Layout>(%s)))"
          arg_name
          arg_name
      | TensorOption -> Printf.sprintf "(%s ? *%s : torch::Tensor())" arg_name arg_name
      | Bool -> "(bool)" ^ arg_name
      | IntList -> Printf.sprintf "torch::IntArrayRef(%s_data, %s_len)" arg_name arg_name
      | IntListOption ->
        Printf.sprintf
          "%s_data == nullptr ? c10::nullopt : \
           c10::optional<torch::IntArrayRef>(torch::IntArrayRef(%s_data, %s_len))"
          arg_name
          arg_name
          arg_name
      | DoubleList ->
        Printf.sprintf "at::ArrayRef<double>(%s_data, %s_len)" arg_name arg_name
      | String -> Printf.sprintf "std::string(%s_ptr, %s_len)" arg_name arg_name
      | TensorOptList ->
        Printf.sprintf "of_carray_tensor_opt(%s_data, %s_len)" arg_name arg_name
      | TensorList -> Printf.sprintf "of_carray_tensor(%s_data, %s_len)" arg_name arg_name
      | TensorOptions ->
        Printf.sprintf
          "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))"
          arg_name
          arg_name
      | Int64Option ->
        Printf.sprintf
          "%s_null ? c10::nullopt : c10::optional<int64_t>(%s_v)"
          arg_name
          arg_name
      | DoubleOption ->
        Printf.sprintf
          "%s_null ? c10::nullopt : c10::optional<double>(%s_v)"
          arg_name
          arg_name
      | ScalarType -> Printf.sprintf "at::ScalarType(%s)" arg_name
      | ScalarTypeOption ->
        Printf.sprintf
          "%s < 0 ? c10::nullopt : c10::optional<at::ScalarType>(at::ScalarType(%s))"
          arg_name
          arg_name
      | Device -> Printf.sprintf "device_of_int(%s)" arg_name
      | _ -> arg_name)
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> Printf.sprintf "torch::%s(%s)" t.name (c_args_list t.args)
    | `method_ ->
      (match t.args with
       | head :: tail ->
         Printf.sprintf "%s->%s(%s)" head.arg_name t.name (c_args_list tail)
       | [] ->
         Printf.failwithf "Method calls should have at least one argument %s" t.name ())

  (* 
  let replace_map =
    Map.of_alist_exn
      (module String)
      [ ("t", "tr")
      ; ("where", "where_")
      ; ("view", "view_")
      ; ("unsafe", "unsafe_")
      ; ("to_device", "to_device_") ]
 *)

  let operator_name t =
    match String.lowercase t.operator_name with
    | "scatter_reduce" ->
      (* scatter_reduce is both an operator name and also obtained from the
         scatter operator when using the reduce overload. *)
      "_scatter_reduce"
    | "scatter_reduce_" -> "_scatter_reduce_"
    | other -> other

  let is_method t =
    List.exists t.args ~f:(fun arg ->
        match arg.arg_name with "self" -> true | _ -> false )

  let go_name name =
    let last_underscore name = Str.string_match (Str.regexp ".*_$") name 0 in
    let words = Str.split (Str.regexp "_") name in
    if last_underscore name then
      let cap_words = List.map words ~f:(fun word -> String.capitalize word) in
      String.concat ~sep:"" cap_words ^ "_"
    else
      let cap_words = List.map words ~f:(fun word -> String.capitalize word) in
      String.concat ~sep:"" cap_words

  let go_variable name =
    let goname = go_name name in
    (* NOTE: Deal with Go namespace conflict *)
    let safe_name =
      match goname with
      | "Var" -> "vari"
      | "Unsafe" -> "unsafety"
      | "Range" -> "rangeVals"
      | _ -> goname
    in
    String.uncapitalize safe_name

  let c_go_args_list t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        let single_param = Printf.sprintf "%s %s" an in
        match arg.arg_type with
        | Bool -> single_param "int32"
        | Layout | LayoutOption -> single_param "int8"
        | Int64 -> single_param "int64"
        | Double -> single_param "float64"
        | Tensor -> single_param "Ctensor"
        | TensorOption -> single_param "Ctensor"
        | Scalar -> single_param "Cscalar"
        | ScalarType | ScalarTypeOption -> single_param "int32"
        | Device -> single_param "int32"
        | String -> single_param "string"
        | IntList | IntListOption -> Printf.sprintf "%sData []int64, %sLen int" an an
        | DoubleList -> Printf.sprintf "%sData []float64, %sLen int" an an
        | TensorOptList -> Printf.sprintf "%sData []Ctensor, %sLen int" an an
        | TensorList -> Printf.sprintf "%sData []Ctensor, %sLen int" an an
        | Int64Option -> Printf.sprintf "%sVal int64, %sNull int" an an
        | DoubleOption -> Printf.sprintf "%sVal float64, %sNull int" an an
        | TensorOptions -> Printf.sprintf "%sKind int32, %sDevice int32" an an
    )
    |> String.concat ~sep:", "

  let c_go_args_list_notype t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        let an = match an with "var" -> "vari" | _ -> an in
        let single_param = Printf.sprintf "%s %s" an in
        match arg.arg_type with
        | Bool -> Printf.sprintf "c%s" an
        | Int64 -> Printf.sprintf "c%s" an
        | Double -> Printf.sprintf "c%s" an
        | Tensor -> Printf.sprintf "%s" an
        | TensorOption -> Printf.sprintf "%s" an
        | Layout | LayoutOption -> Printf.sprintf "c%s" an
        | Scalar -> single_param ""
        | ScalarType | ScalarTypeOption -> Printf.sprintf "c%s" an
        | Device -> Printf.sprintf "c%s" an
        | String -> Printf.sprintf "c%s, c%sLen" an an
        | IntList | IntListOption -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | DoubleList -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | TensorOptList -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | TensorList -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | Int64Option -> Printf.sprintf "c%sVal, c%sNull" an an
        | DoubleOption -> Printf.sprintf "c%sVal, c%sNull" an an
        | TensorOptions -> Printf.sprintf "c%sKind, c%sDevice" an an )
    |> String.concat ~sep:", "

  (* TODO: convert Go pointer to C pointer *)
  let c_go_args_list_body t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        (* let single_param = Printf.sprintf "%s %s" an in *)
        match arg.arg_type with
        | Bool ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | Int64 ->
            Printf.sprintf "\nc%s := *(*C.int64_t)(unsafe.Pointer(&%s))" an an
        | Double ->
            Printf.sprintf "\nc%s := *(*C.double)(unsafe.Pointer(&%s))" an an
        | Tensor -> ""
        | TensorOption -> ""
        | Layout | LayoutOption ->
            Printf.sprintf "\nc%s := *(*C.int8_t)(unsafe.Pointer(&%s))" an an
        | Scalar -> ""
        | ScalarType | ScalarTypeOption ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | Device ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | String ->
            Printf.sprintf
              "\n\
               c%s := C.CString(%s)\n\
               %sLen := len(%s)\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an an an
        | IntList | IntListOption ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*C.int64_t)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | DoubleList ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*C.double)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | TensorOptList ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*Ctensor)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | TensorList ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*Ctensor)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | Int64Option ->
            Printf.sprintf
              "\n\
               c%sVal := *(*C.int64_t)(unsafe.Pointer(&%sVal))\n\
               c%sNull := *(*C.uint8_t)(unsafe.Pointer(&%sNull))"
              an an an an
        | DoubleOption ->
            Printf.sprintf
              "\n\
               c%sVal := *(*C.double)(unsafe.Pointer(&%sVal))\n\
               c%sNull := *(*C.uint8_t)(unsafe.Pointer(&%sNull))"
              an an an an
        | TensorOptions ->
            Printf.sprintf
              "\n\
               c%sKind := *(*C.int)(unsafe.Pointer(&%sKind))\n\
               c%sDevice := *(*C.int)(unsafe.Pointer(&%sDevice))"
              an an an an )
    |> String.concat ~sep:""

  let self_name = "self"

  let self_tensor arg =
    match arg.arg_type with
    | Tensor -> String.( = ) arg.arg_name self_name
    | _ -> false

  (* 
 *   let type_parameters t =
 *     let needs_scalar_parameter =
 *       List.exists t.args ~f:(fun arg ->
 *           match arg.arg_type with Scalar -> true | _ -> false )
 *     in
 *     let needs_type_parameter =
 *       List.exists t.args ~f:(fun arg ->
 *           match arg.arg_type with
 *           | TensorList | TensorOption -> true
 *           | _ -> false )
 *     in
 *     if needs_type_parameter && needs_scalar_parameter then "Tensor, Scalar"
 *     else if needs_type_parameter then "Tensor"
 *     else if needs_scalar_parameter then "Scalar"
 *     else ""
 *  *)
  
  (* 
 *   let go_args_list t =
 *     (* https://ocaml.janestreet.com/ocaml-core/latest/doc/base/Base/List/#val-partition_tf *)
 *     (* TODO. implement special cases - TensorOptions, ... *)
 *     match List.partition_tf t.args ~f:self_tensor with _, args_list ->
 *       args_list
 *  *)

  let is_inplace t =
    match Str.string_match (Str.regexp ".*_$") t.name 0 with
    | true -> true
    | _ -> false

  let go_typed_args_list t =
    let to_string args =
      let args_list =
        List.map args ~f:(fun arg ->
            let go_arg_type =
              match arg.arg_type with
              | Bool -> "bool"
              | Layout | LayoutOption -> "Layout"
              | Int64 -> "int64"
              | Double -> "float64"
              | Tensor -> "*Tensor"
              | TensorOption -> "*Tensor"
              | IntList | IntListOption -> "[]int64"
              | DoubleList -> "[]float64"
              | TensorOptList -> "[]*Tensor"
              | TensorList -> "[]*Tensor"
              | String -> "string"
              (* TODO. Struct{Kind gotch.DType Device gotch.Device} *)
              (* E.g. `type KindDevice struct{}` *)
              | TensorOptions -> "gotch.KindDevice"
              | Scalar -> "*Scalar"
              | ScalarType | ScalarTypeOption -> "gotch.DType"
              | Int64Option -> "[]int64"
              | DoubleOption -> "[]float64"
              | Device -> "gotch.Device"
            in
            match arg.arg_type with
            | TensorOptions ->
                Printf.sprintf "%sKind gotch.DType, %sDevice gotch.Device"
                  (go_variable arg.arg_name) (go_variable arg.arg_name)
            | _ -> Printf.sprintf "%s %s" (go_variable arg.arg_name) go_arg_type
        )
      in
      if is_method t && not (is_inplace t) then
        args_list @ ["del bool"] |> String.concat ~sep:", "
      else args_list |> String.concat ~sep:", "
    in
    (* let self_arg = "self Tensor" in *)
    match List.partition_tf t.args ~f:self_tensor with _, args_list ->
      Printf.sprintf "%s" (to_string args_list)

  let go_notype_args_list t =
    let to_string args =
      let args_list =
        List.map args ~f:(fun arg ->
            match arg.arg_type with
            | TensorOptions ->
                Printf.sprintf "%sKind, %sDevice" (go_variable arg.arg_name)
                  (go_variable arg.arg_name)
            | _ -> Printf.sprintf "%s" (go_variable arg.arg_name) )
      in
      if is_method t && not (is_inplace t) then
        args_list @ ["del"] |> String.concat ~sep:", "
      else args_list |> String.concat ~sep:", "
    in
    match List.partition_tf t.args ~f:self_tensor with _, args_list ->
      Printf.sprintf "%s" (to_string args_list)

  let go_return_type t ~fallible =
    (* printf "t name: %s\n" t.name ; *)
    let returns =
      match t.returns with
      | `nothing -> None
      | `fixed 1 -> Some "retVal *Tensor"
      | `fixed v ->
          List.init v ~f:(fun i -> Printf.sprintf "retVal%d *Tensor" i)
          |> String.concat ~sep:", " 
          |> Printf.sprintf "%s"
          |> Option.some
      | `dynamic -> Some "retVal []*Tensor"
      | `bool -> Some "retVal bool"
      | `int64_t -> Some "retVal int64"
      | `double -> Some "retVal float64"
    in
    match returns with
    | None -> if fallible then Printf.sprintf "err error" else Printf.sprintf ""
    | Some returns -> 
      if is_inplace t then
        if fallible then Printf.sprintf "err error" else Printf.sprintf ""
      else if fallible then Printf.sprintf "%s, err error" returns
      else Printf.sprintf "%s" returns

  let go_return_notype t ~fallible =
    let returns =
      match t.returns with
      | `nothing -> None
      | `fixed 1 -> Some "retVal"
      | `fixed v ->
          List.init v ~f:(fun i -> Printf.sprintf "retVal%d" i)
          |> String.concat ~sep:", " 
          |> Printf.sprintf "%s"
          |> Option.some
      | `dynamic -> Some "retVal"
      | `bool -> Some "retVal"
      | `int64_t -> Some "retVal"
      | `double -> Some "retVal"
    in
    match returns with
    | None -> if fallible then Printf.sprintf "err" else Printf.sprintf ""
    | Some returns ->
      if is_inplace t then
        if fallible then Printf.sprintf "err" else Printf.sprintf ""
      else if fallible then Printf.sprintf "%s, err" returns
      else Printf.sprintf "%s" returns

  let go_binding_args t =
    List.map t.args ~f:(fun arg ->
        let name = go_variable arg.arg_name in
        match arg.arg_type with
        | Tensor ->
            if String.( = ) name "self" then "ts.ctensor"
            else Printf.sprintf "%s.ctensor" name
        | Scalar -> Printf.sprintf "%s.cscalar" name
        | Bool -> Printf.sprintf "c%s" name
        | ScalarType | ScalarTypeOption -> Printf.sprintf "%s.CInt()" name
        | Device -> Printf.sprintf "%s.CInt()" name
        | TensorOptions ->
            Printf.sprintf "%sKind.CInt(), %sDevice.CInt()" name name
        | String -> Printf.sprintf "%s" name
        | IntList -> Printf.sprintf "%s, %sLen" name name
        | IntListOption -> Printf.sprintf "%s, %sLen" name name
        | DoubleList -> Printf.sprintf "%s, %sLen" name name
        | TensorList -> Printf.sprintf "c%s, len(c%s)" name name
        | TensorOptList -> Printf.sprintf "c%s, len(c%s)" name name
        | Int64Option -> Printf.sprintf "c%sVal, c%sNull" name name
        | DoubleOption -> Printf.sprintf "c%sVal, c%sNull" name name
        | TensorOption -> Printf.sprintf "%s.ctensor" name
        | Layout -> Printf.sprintf "int8(%s)" name
        | LayoutOption -> Printf.sprintf "int8(%s)" name 
        | _ -> name )
    |> String.concat ~sep:", "

  let go_binding_body t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        match arg.arg_type with
        | Bool ->
            Printf.sprintf "c%s := int32(0)\n if %s { c%s = int32(1) }\n" an an
              an
        | Int64 -> ""
        | Double -> ""
        | Tensor -> ""
        | TensorOption -> ""
        | Scalar -> ""
        | ScalarType | ScalarTypeOption -> ""
        | Device -> ""
        | String -> ""
        | IntList | IntListOption -> Printf.sprintf "%sLen := len(%s)\n" an an
        | DoubleList -> Printf.sprintf "%sLen := len(%s)\n" an an
        | Int64Option ->
            Printf.sprintf
              "var c%sVal int64 = 0\n\
              \ var c%sNull int = 1\n\
              \ if len(%s) > 0 {\n\
              \ c%sVal = %s[0]\n\
              \ c%sNull = 0\n\
              \ }\n"
              an an an an an an
        | DoubleOption ->
            Printf.sprintf
              "var c%sVal float64 = 0.0\n\
              \ var c%sNull int = 1\n\
              \ if len(%s) > 0 {\n\
              \ c%sVal = %s[0]\n\
              \ c%sNull = 0\n\
              \ }\n"
              an an an an an an
        | TensorOptList ->
            Printf.sprintf
              "var c%s []lib.Ctensor\n\
              \  for _, t := range %s {c%s = append(c%s, t.ctensor)}\n"
              an an an an
        | TensorList ->
            Printf.sprintf
              "var c%s []lib.Ctensor\n\
              \  for _, t := range %s {c%s = append(c%s, t.ctensor)}\n"
              an an an an
        | Layout | LayoutOption -> "" 
        | TensorOptions -> "" )
    |> String.concat ~sep:""
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l -> String.length l > 0 && Char.( = ) l.[0] '-')
    |> List.concat_map ~f:(fun lines ->
         Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list)
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun yaml ->
    let map = extract_map yaml in
    let name = Map.find_exn map "name" |> extract_string in
    let operator_name = Map.find_exn map "operator_name" |> extract_string in
    let overload_name = Map.find_exn map "overload_name" |> extract_string in
    let deprecated = Map.find_exn map "deprecated" |> extract_bool in
    let method_of =
      Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
    in
    let arguments = Map.find_exn map "arguments" |> extract_list in
    let returns =
      let is_tensor returns =
        let returns = extract_map returns in
        let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
        String.( = ) return_type "at::Tensor"
      in
      let returns = Map.find_exn map "returns" |> extract_list in
      if List.is_empty returns
      then Some `nothing
      else if List.for_all returns ~f:is_tensor
      then Some (`fixed (List.length returns))
      else (
        match returns with
        | [ returns ] ->
          let return_type =
            Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
          in
          (match return_type with
           | "bool" -> Some `bool
           | "int64_t" -> Some `int64_t
           | "double" -> Some `double
           | "at::TensorList" | "dynamic_type: const c10::List<c10::optional<Tensor>> &"
             -> Some `dynamic
           | _ -> None)
        | [] | _ :: _ :: _ -> None)
    in
    let kind =
      if List.exists method_of ~f:(String.( = ) "namespace")
      then Some `function_
      else if List.exists method_of ~f:(String.( = ) "Tensor")
      then Some `method_
      else None
    in
    if (not deprecated)
       && (not
             (List.exists excluded_prefixes ~f:(fun prefix ->
                String.is_prefix name ~prefix)))
       && (not
             (List.exists excluded_suffixes ~f:(fun suffix ->
                String.is_suffix name ~suffix)))
       && not (Set.mem excluded_functions name)
    then
      Option.both returns kind
      |> Option.bind ~f:(fun (returns, kind) ->
           try
             let args ~with_optional_scalar_args =
               List.filter_map arguments ~f:(fun arg ->
                 let arg = extract_map arg in
                 let arg_name = Map.find_exn arg "name" |> extract_string in
                 let arg_type = Map.find_exn arg "dynamic_type" |> extract_string in
                 let is_nullable =
                   Map.find arg "is_nullable"
                   |> Option.value_map ~default:false ~f:extract_bool
                 in
                 let default_value =
                   Map.find arg "default" |> Option.map ~f:extract_string
                 in
                 match Func.arg_type_of_string arg_type ~is_nullable with
                 | Some Scalar when Option.is_some default_value && not is_nullable ->
                   if with_optional_scalar_args
                   then Some { Func.arg_name; arg_type = Scalar; default_value }
                   else None
                 | Some TensorOptions
                   when Option.is_some default_value && Set.mem no_tensor_options name ->
                   None
                 | Some arg_type ->
                   let arg_name =
                     match arg_name, arg_type with
                     | "self", Scalar -> "self_scalar"
                     | _, _ -> arg_name
                   in
                   Some { Func.arg_name; arg_type; default_value }
                 | None ->
                   if Option.is_some default_value then None else raise Not_a_simple_arg)
             in
             let args =
               args ~with_optional_scalar_args:(Set.mem with_optional_scalar_args name)
             in
             Some [ { Func.name; operator_name; overload_name; args; returns; kind } ]
           with
           | Not_a_simple_arg -> None)
    else None)

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line ;
      Out_channel.output_char out_channel '\n' )
    s

let print_inline out_channel s =
  Printf.ksprintf (fun msg -> Out_channel.output_string out_channel msg) s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp.h") ~f:(fun out_cpp ->
    Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
        let pc s = p out_cpp s in
        let ph s = p out_h s in
        pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        pc "";
        ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        ph "";
        Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
            let c_typed_args_list = Func.c_typed_args_list func in
            match func.returns with
            | `nothing ->
              pc "void atg_%s(%s) {" exported_name c_typed_args_list;
              pc "  PROTECT(";
              pc "    %s;" (Func.c_call func);
              pc "  )";
              pc "}";
              pc "";
              ph "void atg_%s(%s);" exported_name c_typed_args_list
            | `fixed ntensors ->
              pc "void atg_%s(tensor *out__, %s) {" exported_name c_typed_args_list;
              pc "  PROTECT(";
              pc "    auto outputs__ = %s;" (Func.c_call func);
              if ntensors = 1
              then pc "    out__[0] = new torch::Tensor(outputs__);"
              else
                for i = 0 to ntensors - 1 do
                  pc "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i
                done;
              pc "  )";
              pc "}";
              pc "";
              ph "void atg_%s(tensor *, %s);" exported_name c_typed_args_list
            | `dynamic ->
              pc "tensor *atg_%s(%s) {" exported_name c_typed_args_list;
              pc "  PROTECT(";
              pc "    auto outputs__ = %s;" (Func.c_call func);
              (* the returned type is a C++ vector of tensors *)
              pc "    int sz = outputs__.size();";
              pc
                "    torch::Tensor **out__ = (torch::Tensor**)malloc((sz + 1) * \
                 sizeof(torch::Tensor*));";
              pc "    for (int i = 0; i < sz; ++i)";
              pc "      out__[i] = new torch::Tensor(outputs__[i]);";
              pc "    out__[sz] = nullptr;";
              pc "    return out__;";
              pc "  )";
              pc "  return nullptr;";
              pc "}";
              pc "";
              ph "tensor *atg_%s(%s);" exported_name c_typed_args_list
            | (`bool | `int64_t | `double) as returns ->
              let c_type =
                match returns with
                | `bool -> "int"
                | `int64_t -> "int64_t"
                | `double -> "double"
              in
              pc "%s atg_%s(%s) {" c_type exported_name c_typed_args_list;
              pc "  PROTECT(";
              pc "    return %s;" (Func.c_call func);
              pc "  )";
              pc "  return 0;";
              pc "}";
              pc "";
              ph "%s atg_%s(%s);" c_type exported_name c_typed_args_list)))

let write_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = print_inline out_ml s in
      pm "package ts" ;
      pm "\n\n" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "\n\n" ;
      pm "// #include \"stdlib.h\"\n" ;
      pm "import \"C\"" ;
      pm "" ;
      pm "\n\n" ;
      pm "import(\n" ;
      pm "  \"unsafe\"\n" ;
      pm "  \"fmt\"\n" ;
      pm "\n" ;
      pm "  \"github.com/sugarme/gotch\"\n" ;
      pm "  lib \"github.com/sugarme/gotch/libtch\"\n" ;
      pm ")" ;
      pm "\n\n" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data: (func : Func.t) ->
          let is_method = Func.is_method func in
          let is_inplace = Func.is_inplace func in
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let gofunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          let cfunc_name = "lib.Atg" ^ gofunc_name in
          let go_args_list = Func.go_typed_args_list func in
          (* NOTE. temporarily excluding these functions as not implemented at FFI *)
          (* TODO. implement multiple tensors return function []Tensor *)
          let excluded_funcs =
            [ "Chunk"
            ; "AlignTensors"
            ; "BroadcastTensors"
            ; "Meshgrid"
            ; "MeshgridIndexing"
            ; "_ToCpu"
            ; "NonzeroNumpy"
            ; "Split"
            ; "SplitWithSizes"
            ; "Unbind"
            ; "Where"
            ; "Atleast1d1"
            ; "Atleast2d1"
            ; "Atleast3d1"
            ; "Dequantize1"
            ; "QuantizePerTensor1"
            ; "UnsafeChunk"
            ; "UnsafeSplit"
            ; "UnsafeSplitWithSizes"
            ; "AlignTensors"
            ; "UnflattenDenseTensors"
            ; "TensorSplit"
            ; "TensorSplitIndices"
            ; "TensorSplitTensorIndicesOrSections"
            ; "QuantizePerTensorTensors"
            ; "Dsplit"
            ; "DsplitArray"
            ; "Hsplit"
            ; "HsplitArray"
            ; "Vsplit"
            ; "VsplitArray"
            ; "DequantizeTensors"
            ; "Atleast1dSequence"
            ; "Atleast2dSequence"
            ; "Atleast3dSequence"
            ; "Index"
            ; "IndexPut"
            ; "IndexPut_"
            ; "_IndexPutImpl_" ]
          in
          if
            List.exists excluded_funcs ~f:(fun name ->
                String.( = ) name gofunc_name )
          then pm ""
          else
            match func.returns with
            | `nothing -> pm ""; (* TODO. implement manually*)
            | `dynamic ->  pm ""; (* TODO. implement manually*)
            | `fixed 1 ->
                pm "\n" ;

                pm "// func.returns = `fixed 1`: \n";
                pm "// --------------------------\n";
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "  if del { defer ts.MustDrop() }\n" ;
                pm "  ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "  %s(ptr, %s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    err = fmt.Errorf(\"%s() failed: %%w\", err)\n" gofunc_name;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                (* NOTE. if in_place method, no retVal return *)
                if not (Func.is_inplace func) then
                  pm "  retVal = newTensor(*ptr, \"%s\")\n" gofunc_name
                else pm "  ts.ctensor = *ptr\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"

                
            | `fixed ntensors ->
                pm "\n" ;

                pm "// func.returns = `fixed ntensors`: \n";
                pm "// ---------------------------------\n";
                
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "  if del { defer ts.MustDrop() }\n" ;
                for i = 0 to ntensors - 1 do
                  (* pc "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i *)
                  if i = 0 then
                    pm
                      "  ctensorPtr0 := \
                       (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))\n"
                  else
                    pm
                      "  ctensorPtr%d := \
                       (*lib.Ctensor)(unsafe.Pointer(uintptr(unsafe.Pointer(ctensorPtr%d)) \
                       + unsafe.Sizeof(ctensorPtr0)))\n"
                      i (i - 1)
                done ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "  %s(ctensorPtr0, %s)\n" cfunc_name
                  (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    err = fmt.Errorf(\"%s() failed: %%w\", err)\n" gofunc_name;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                (* NOTE. if in_place method, no retVal return *)
                if not (Func.is_inplace func) then
                  for i = 0 to ntensors - 1 do
                    pm "  retVal%d = newTensor(*ctensorPtr%d, \"%s_%d\")\n" i i gofunc_name i
                  done
                else pm "  ts.ctensor = *ptr\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"
            | `bool ->
                pm "\n" ;

                pm "// func.returns = `bool`: \n";
                pm "// --------------------------\n";

                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "  if del { defer ts.MustDrop() }\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "  retVal = %s(%s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    err = fmt.Errorf(\"%s() failed: %%w\", err)\n" gofunc_name;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"
            | `int64_t ->
                pm "\n" ;

                pm "// func.returns = `int64`: \n";
                pm "// --------------------------\n";

                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "  if del { defer ts.MustDrop() }\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "  retVal = %s(%s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    err = fmt.Errorf(\"%s() failed: %%w\", err)\n" gofunc_name;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"
            | `double ->

                pm "// func.returns = `double `: \n";
                pm "// --------------------------\n";

                pm "\n" ;
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "if del { defer ts.MustDrop() }\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "  retVal = %s(%s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    err = fmt.Errorf(\"%s() failed: %%w\", err)\n" gofunc_name;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n" ) ;
      pm "// End of implementing Tensor ================================= \n"
  )

let write_must_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = print_inline out_ml s in
      pm "package ts" ;
      pm "\n\n" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "\n\n" ;
      pm "import(\n" ;
      pm "  \"log\"\n" ;
      pm "\n" ;
      pm "  \"github.com/sugarme/gotch\"\n" ;
      pm ")" ;
      pm "\n\n" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:(func: Func.t) ->
          let is_method = Func.is_method func in
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let gofunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          let go_args_list = Func.go_typed_args_list func in
          let go_args_list_notype = Func.go_notype_args_list func in
          (* NOTE. temporarily excluding these functions as not implemented at FFI *)
          let excluded_funcs =
            [ "Chunk"
            ; "AlignTensors"
            ; "BroadcastTensors"
            ; "Meshgrid"
            ; "MeshgridIndexing"
            ; "_ToCpu"
            ; "NonzeroNumpy"
            ; "Split"
            ; "SplitWithSizes"
            ; "Unbind"
            ; "Where"
            ; "Atleast1d1"
            ; "Atleast2d1"
            ; "Atleast3d1"
            ; "Dequantize1"
            ; "QuantizePerTensor1"
            ; "UnsafeChunk"
            ; "UnsafeSplit"
            ; "UnsafeSplitWithSizes"
            ; "AlignTensors"
            ; "UnflattenDenseTensors"
            ; "TensorSplit"
            ; "TensorSplitIndices"
            ; "TensorSplitTensorIndicesOrSections"
            ; "QuantizePerTensorTensors"
            ; "Dsplit"
            ; "DsplitArray"
            ; "Hsplit"
            ; "HsplitArray"
            ; "Vsplit"
            ; "VsplitArray"
            ; "DequantizeTensors"
            ; "Atleast1dSequence"
            ; "Atleast2dSequence"
            ; "Atleast3dSequence"
            ; "Index"
            ; "IndexPut"
            ; "IndexPut_"
            ; "_IndexPutImpl_" ]
          in
          if
            List.exists excluded_funcs ~f:(fun name ->
                String.( = ) name gofunc_name )
          then pm ""
          else
            match func.returns with
            | `nothing -> pm "" (*TODO: implement manually*)
            | `dynamic ->  pm ""; (* TODO. implement manually*)
            | `fixed 1 ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                (* NOTE. No return retVal for in_place method *)
                if Func.is_inplace func then
                  if is_method then
                    pm "  err := ts.%s(%s)\n" gofunc_name go_args_list_notype
                  else pm "  err := %s(%s)\n" gofunc_name go_args_list_notype
                else if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `fixed _ ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                (* NOTE. No return retVal for in_place method *)
                if Func.is_inplace func then
                  if is_method then
                    pm "  err := ts.%s(%s)\n" gofunc_name go_args_list_notype
                  else pm "  err := %s(%s)\n" gofunc_name go_args_list_notype
                else if is_method then
                  pm "  %s, err := ts.%s(%s)\n"
                    (Func.go_return_notype func ~fallible:false)
                    gofunc_name go_args_list_notype
                else
                  pm "  %s, err := %s(%s)\n"
                    (Func.go_return_notype func ~fallible:false)
                    gofunc_name go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `bool ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `int64_t ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `double ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n" ) ;
      pm "// End of implementing Tensor ================================= \n"
  )

let write_ffi funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = p out_ml s in
      pm "package libtch" ;
      pm "" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "" ;
      pm "//#include \"stdbool.h\" " ;
      pm "//#include \"torch_api.h\" " ;
      pm "import \"C\"" ;
      pm "" ;
      pm "import \"unsafe\"" ;
      pm "" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data: (func : Func.t) ->
          (* let is_method = *)
          (* match func.Func.kind with `method_ -> true | `function_ -> false *)
          (* in *)
          (* let is_inplace = *)
          (* Func.is_inplace func *)
          (* 
 *             match exported_name with
 *             | "add_1" -> true
 *             | "sub_1" -> true
 *             | "div_1" -> true
 *             | "mul_1" -> true
 *             | _ -> false
 *  *)
          (* in *)
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let ffifunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          match func.Func.returns with
          | `nothing -> pm "" (*TODO.*)
          | `fixed _ ->
              pm "func Atg%s(ptr *Ctensor, %s){%s \n\tC.atg_%s(ptr, %s)\n}"
                ffifunc_name (Func.c_go_args_list func)
                (Func.c_go_args_list_body func)
                exported_name
                (Func.c_go_args_list_notype func)
          | `dynamic -> pm "" (*TODO.*)
          | `bool ->
              pm "func Atg%s(%s) bool{%s" ffifunc_name
                (Func.c_go_args_list func)
                (Func.c_go_args_list_body func) ;
              pm "\t cResult := C.atg_%s(%s)" exported_name
                (Func.c_go_args_list_notype func) ;
              pm "\t cbool := (int)(cResult)" ;
              pm "\t if cbool == 1{return true}" ;
              pm "\t return false" ;
              pm "}"
          | `int64_t ->
              pm "func Atg%s(%s) int64{%s" ffifunc_name
                (Func.c_go_args_list func)
                (Func.c_go_args_list_body func) ;
              pm "\t cResult := C.atg_%s(%s)" exported_name
                (Func.c_go_args_list_notype func) ;
              pm "\t return *(*int64)(unsafe.Pointer(&cResult))" ;
              pm "}"
          | `double ->
              pm "func Atg%s(%s) float64{%s" ffifunc_name
                (Func.c_go_args_list func)
                (Func.c_go_args_list_body func) ;
              pm "\t cResult := C.atg_%s(%s)" exported_name
                (Func.c_go_args_list_notype func) ;
              pm "\t return *(*float64)(unsafe.Pointer(&cResult))" ;
              pm "}"
          (* TODO: need more implement here *)
          (* pm "func Atg%s(%s)(retValPtr *Ctensor)" *)
          (* (Func.go_name exported_name) *)
          (* (Func.c_go_args_list func)  *) ) )

let methods =
  let c name args =
    { Func.name
    ; operator_name = name
    ; overload_name = ""
    ; args
    ; returns = `fixed 1
    ; kind = `method_
    }
  in
  let ca arg_name arg_type = { Func.arg_name; arg_type; default_value = None } in
  [ c "grad" [ ca "self" Tensor ]
  ; c "set_requires_grad" [ ca "self" Tensor; ca "r" Bool ]
  ; c "toType" [ ca "self" Tensor; ca "scalar_type" ScalarType ]
  ; c "to" [ ca "self" Tensor; ca "device" Device ]
  ]

let run ~yaml_filename ~cpp_filename ~ffi_filename ~must_wrapper_filename
    ~wrapper_filename =
  let funcs = read_yaml yaml_filename |> List.concat in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> Func.operator_name func, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
         match funcs with
         | [] -> assert false
         | [ func ] -> [ name, func ]
         | funcs ->
           let has_empty_overload =
             List.exists funcs ~f:(fun (func : Func.t) ->
               String.is_empty func.overload_name)
           in
           List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
             match Int.compare (String.length f1.name) (String.length f2.name) with
             | 0 -> Int.compare (List.length f1.args) (List.length f2.args)
             | cmp -> cmp)
           |> List.mapi ~f:(fun index (func : Func.t) ->
                let operator_name = Func.operator_name func in
                let overload_name = String.lowercase func.overload_name in
                let name =
                  if String.is_empty overload_name || (index = 0 && not has_empty_overload)
                  then operator_name
                  else if String.is_suffix operator_name ~suffix:"_"
                  then operator_name ^ overload_name ^ "_"
                  else operator_name ^ "_" ^ overload_name
                in
                name, func))
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename;
  write_ffi funcs ffi_filename ;
  write_must_wrapper funcs must_wrapper_filename ;
  write_wrapper funcs wrapper_filename


let () =
  run ~yaml_filename:"gen/pytorch/Declarations-v2.10.0.yaml"
    ~cpp_filename:"libtch/torch_api_generated"
    ~ffi_filename:"libtch/c-generated.go"
    ~must_wrapper_filename:"ts/must-tensor-generated.go"
    ~wrapper_filename:"ts/tensor-generated.go"
