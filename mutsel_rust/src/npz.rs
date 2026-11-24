//! Writing of numpy npz files with f64 arrays in C order ('<f8', without compression).

use std::{io::Write, path::Path};

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

#[cfg(test)]
mod tests {
    use std::io::Read;

    use phylo_grad::nalgebra::SMatrix;

    use super::*;

    /// Shape and data of an npy array as written by npy_bytes
    fn parse_npy(bytes: &[u8]) -> (String, Vec<f64>) {
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': ") && header.ends_with('\n'));
        let shape = header[header.find('(').unwrap()..=header.find(')').unwrap()].to_string();
        let data = bytes[10 + header_len..].chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
        (shape, data)
    }

    #[test]
    fn npz_contains_the_arrays() {
        let dir = std::env::temp_dir().join(format!("mutsel_npz_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.npz");
        let matrix = SMatrix::<f64, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, -6.5);
        let arrays = [
            ("scalar", Array::scalar(1.5), "()"),
            ("vector", Array::vector(vec![1.0, f64::NEG_INFINITY, 3.0]), "(3,)"),
            ("matrix", Array::from_matrix(&matrix), "(2, 3)"),
            ("empty", Array::new(&[0, 20], vec![]), "(0, 20)"),
        ];
        write_npz(&path, &arrays.iter().map(|(n, a, _)| (*n, a)).collect::<Vec<_>>()).unwrap();
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(archive.len(), arrays.len());
        for (name, array, shape) in &arrays {
            let mut bytes = Vec::new();
            archive.by_name(&format!("{}.npy", name)).unwrap().read_to_end(&mut bytes).unwrap();
            assert_eq!(parse_npy(&bytes), (shape.to_string(), array.data.clone()), "{}", name);
        }
        // row-major
        assert_eq!(arrays[2].1.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, -6.5]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
