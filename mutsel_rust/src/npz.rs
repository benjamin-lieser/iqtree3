//! Reading and writing of numpy npz files with f64 arrays in C order.
//!
//! Arrays are written as '<f8' without compression. Reading also accepts '<f4', '<i4' and '<i8' arrays
//! (converted to f64) and compressed files (np.savez_compressed).

use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use phylo_grad::nalgebra::{SMatrix, SVector};

use crate::Result;

/// An array with its shape, the data in C order (row-major)
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn new(shape: &[usize], data: Vec<f64>) -> Array {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape {:?} does not match the data", shape);
        Array { shape: shape.to_vec(), data }
    }

    pub fn scalar(value: f64) -> Array {
        Array::new(&[], vec![value])
    }

    pub fn vector(data: Vec<f64>) -> Array {
        Array::new(&[data.len()], data)
    }

    /// [R, C] array of a matrix
    pub fn from_matrix<const R: usize, const C: usize>(matrix: &SMatrix<f64, R, C>) -> Array {
        Array::new(&[R, C], matrix.transpose().as_slice().to_vec())
    }

    /// [n, N] array with one row per vector
    pub fn from_rows<const N: usize>(rows: &[SVector<f64, N>]) -> Array {
        Array::new(&[rows.len(), N], rows.iter().flat_map(|row| row.iter().copied()).collect())
    }

    pub fn to_scalar(&self) -> Result<f64> {
        if self.data.len() != 1 {
            return Err(format!("expected a scalar, got shape {:?}", self.shape).into());
        }
        Ok(self.data[0])
    }

    pub fn to_matrix<const R: usize, const C: usize>(&self) -> Result<SMatrix<f64, R, C>> {
        if self.shape != [R, C] {
            return Err(format!("expected shape [{}, {}], got {:?}", R, C, self.shape).into());
        }
        Ok(SMatrix::<f64, R, C>::from_row_slice(&self.data))
    }

    /// The rows of an array whose last dimension is N, over all other dimensions in C order
    pub fn rows<const N: usize>(&self) -> Result<Vec<SVector<f64, N>>> {
        if self.shape.last() != Some(&N) {
            return Err(format!("expected the last dimension {}, got shape {:?}", N, self.shape).into());
        }
        Ok(self.data.chunks_exact(N).map(SVector::<f64, N>::from_column_slice).collect())
    }
}

/// The arrays of an npz file
pub struct Npz {
    path: PathBuf,
    arrays: HashMap<String, Array>,
}

impl Npz {
    pub fn read(path: &Path) -> Result<Npz> {
        let file = std::fs::File::open(path).map_err(|e| format!("Could not open {}: {}", path.display(), e))?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut arrays = HashMap::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            let name = entry.name()?.trim_end_matches(".npy").to_string();
            let mut bytes = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut bytes)?;
            let array = parse_npy(&bytes).map_err(|e| format!("{} in {}: {}", name, path.display(), e))?;
            arrays.insert(name, array);
        }
        Ok(Npz { path: path.to_path_buf(), arrays })
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.arrays
            .get(name)
            .ok_or_else(|| format!("{} not found in {}", name, self.path.display()).into())
    }

    pub fn get_opt(&self, name: &str) -> Option<&Array> {
        self.arrays.get(name)
    }
}

pub fn write_npz(path: &Path, arrays: &[(&str, &Array)]) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|e| format!("Could not create {}: {}", path.display(), e))?;
    let mut writer = zip::ZipWriter::new(std::io::BufWriter::new(file));
    for (name, array) in arrays {
        let npy = npy_bytes(array);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(npy.len() as u64 >= u32::MAX as u64);
        writer.start_file(format!("{}.npy", name), options)?;
        writer.write_all(&npy)?;
    }
    writer.finish()?.flush()?;
    Ok(())
}

const MAGIC: &[u8] = b"\x93NUMPY";

fn npy_bytes(array: &Array) -> Vec<u8> {
    let shape = match array.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", array.shape[0]),
        _ => format!("({})", array.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}", shape);
    // Version 1.0 with a 2 byte header length, padded so that the data starts at a multiple of 64 bytes
    let unpadded = MAGIC.len() + 2 + 2 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');

    let mut bytes = Vec::with_capacity(MAGIC.len() + 4 + header.len() + 8 * array.data.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in &array.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn parse_npy(bytes: &[u8]) -> std::result::Result<Array, String> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not an npy array".to_string());
    }
    let (header_len, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
        version => return Err(format!("npy version {} is not supported", version)),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).map_err(|e| e.to_string())?;
    let field = |key: &str| -> std::result::Result<&str, String> {
        let start = header.find(&format!("'{}':", key)).ok_or(format!("no {} in the header", key))? + key.len() + 3;
        Ok(header[start..].trim_start())
    };
    let descr = field("descr")?;
    let descr = &descr[1..descr[1..].find('\'').ok_or("invalid descr")? + 1];
    if field("fortran_order")?.starts_with("True") {
        return Err("Fortran order is not supported".to_string());
    }
    let shape = field("shape")?;
    let shape: Vec<usize> = shape[1..shape.find(')').ok_or("invalid shape")?]
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<std::result::Result<_, _>>()?;

    let data = &bytes[header_start + header_len..];
    let count: usize = shape.iter().product();
    let convert = |size: usize, f: &dyn Fn(&[u8]) -> f64| -> std::result::Result<Vec<f64>, String> {
        if data.len() < count * size {
            return Err("the data is shorter than the shape".to_string());
        }
        Ok(data[..count * size].chunks_exact(size).map(f).collect())
    };
    let values = match descr {
        "<f8" => convert(8, &|b| f64::from_le_bytes(b.try_into().unwrap()))?,
        "<f4" => convert(4, &|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)?,
        "<i8" => convert(8, &|b| i64::from_le_bytes(b.try_into().unwrap()) as f64)?,
        "<i4" => convert(4, &|b| i32::from_le_bytes(b.try_into().unwrap()) as f64)?,
        _ => return Err(format!("dtype {} is not supported", descr)),
    };
    Ok(Array { shape, data: values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npz_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mutsel_npz_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.npz");
        let matrix = SMatrix::<f64, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, -6.5);
        let arrays = [
            ("scalar", Array::scalar(1.5)),
            ("vector", Array::vector(vec![1.0, f64::NEG_INFINITY, 3.0])),
            ("matrix", Array::from_matrix(&matrix)),
            ("empty", Array::new(&[0, 20], vec![])),
            ("cube", Array::new(&[2, 1, 3], (0..6).map(|i| i as f64).collect())),
        ];
        write_npz(&path, &arrays.iter().map(|(n, a)| (*n, a)).collect::<Vec<_>>()).unwrap();
        let npz = Npz::read(&path).unwrap();
        for (name, array) in &arrays {
            assert_eq!(npz.get(name).unwrap(), array, "{}", name);
        }
        assert_eq!(npz.get("matrix").unwrap().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, -6.5]);
        assert_eq!(npz.get("matrix").unwrap().to_matrix::<2, 3>().unwrap(), matrix);
        assert!(npz.get("missing").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parses_numpy_headers() {
        // Header as written by numpy for np.arange(6, dtype=np.int32).reshape(2, 3)
        let header = "{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }";
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for i in 0..6i32 {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(array.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
