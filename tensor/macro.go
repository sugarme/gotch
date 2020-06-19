package tensor

// TODO: implement tensor.From macro
/*
 * macro_rules! from_tensor {
 *     ($typ:ident, $zero:expr, $kind:ident) => {
 *         impl From<&Tensor> for Vec<$typ> {
 *             fn from(tensor: &Tensor) -> Vec<$typ> {
 *                 let numel = tensor.numel();
 *                 let mut vec = vec![$zero; numel as usize];
 *                 tensor.to_kind(Kind::$kind).copy_data(&mut vec, numel);
 *                 vec
 *             }
 *         }
 *
 *         impl From<&Tensor> for Vec<Vec<$typ>> {
 *             fn from(tensor: &Tensor) -> Vec<Vec<$typ>> {
 *                 let first_dim = tensor.size()[0];
 *                 (0..first_dim)
 *                     .map(|i| Vec::<$typ>::from(tensor.get(i)))
 *                     .collect()
 *             }
 *         }
 *
 *         impl From<&Tensor> for Vec<Vec<Vec<$typ>>> {
 *             fn from(tensor: &Tensor) -> Vec<Vec<Vec<$typ>>> {
 *                 let first_dim = tensor.size()[0];
 *                 (0..first_dim)
 *                     .map(|i| Vec::<Vec<$typ>>::from(tensor.get(i)))
 *                     .collect()
 *             }
 *         }
 *
 *         impl From<&Tensor> for $typ {
 *             fn from(tensor: &Tensor) -> $typ {
 *                 let numel = tensor.numel();
 *                 if numel != 1 {
 *                     panic!("expected exactly one element, got {}", numel)
 *                 }
 *                 Vec::from(tensor)[0]
 *             }
 *         }
 *
 *         impl From<Tensor> for Vec<$typ> {
 *             fn from(tensor: Tensor) -> Vec<$typ> {
 *                 Vec::<$typ>::from(&tensor)
 *             }
 *         }
 *
 *         impl From<Tensor> for Vec<Vec<$typ>> {
 *             fn from(tensor: Tensor) -> Vec<Vec<$typ>> {
 *                 Vec::<Vec<$typ>>::from(&tensor)
 *             }
 *         }
 *
 *         impl From<Tensor> for Vec<Vec<Vec<$typ>>> {
 *             fn from(tensor: Tensor) -> Vec<Vec<Vec<$typ>>> {
 *                 Vec::<Vec<Vec<$typ>>>::from(&tensor)
 *             }
 *         }
 *
 *         impl From<Tensor> for $typ {
 *             fn from(tensor: Tensor) -> $typ {
 *                 $typ::from(&tensor)
 *             }
 *         }
 *     };
 * } */
