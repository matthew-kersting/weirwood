//! UBJSON parser for XGBoost `.ubj` model files.
//!
//! XGBoost follows the UBJSON specification, which mandates **big-endian** byte
//! order for all multi-byte numeric types. This parser implements that format.
//!
//! The output is a `serde_json::Value`, which is then deserialized through the
//! same `RawModel` path used by the JSON loader.

use serde_json::{Map, Number, Value};

use crate::Error;

/// Parse XGBoost UBJ bytes into a `serde_json::Value`.
pub(crate) fn parse(data: &[u8]) -> Result<Value, Error> {
    let mut pos: usize = 0usize;
    let value: Value = parse_value(data, &mut pos)?;
    Ok(value)
}

// ---------------------------------------------------------------------------
// Primitive readers (big-endian, per the UBJSON specification)
// ---------------------------------------------------------------------------

fn read_byte(data: &[u8], pos: &mut usize) -> Result<u8, Error> {
    if *pos >= data.len() {
        return Err(Error::Format("unexpected end of UBJ data".into()));
    }
    let b: u8 = data[*pos];
    *pos += 1;
    Ok(b)
}

fn read_n<const N: usize>(data: &[u8], pos: &mut usize) -> Result<[u8; N], Error> {
    if *pos + N > data.len() {
        return Err(Error::Format("unexpected end of UBJ data".into()));
    }
    let arr: [u8; N] = data[*pos..*pos + N].try_into().unwrap();
    *pos += N;
    Ok(arr)
}

fn read_i8(data: &[u8], pos: &mut usize) -> Result<i8, Error> {
    Ok(read_byte(data, pos)? as i8)
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, Error> {
    read_byte(data, pos)
}

fn read_i16(data: &[u8], pos: &mut usize) -> Result<i16, Error> {
    Ok(i16::from_be_bytes(read_n::<2>(data, pos)?))
}

fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32, Error> {
    Ok(i32::from_be_bytes(read_n::<4>(data, pos)?))
}

fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64, Error> {
    Ok(i64::from_be_bytes(read_n::<8>(data, pos)?))
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, Error> {
    Ok(f32::from_be_bytes(read_n::<4>(data, pos)?))
}

fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64, Error> {
    Ok(f64::from_be_bytes(read_n::<8>(data, pos)?))
}

/// Read a UBJSON string: a typed-integer length followed by UTF-8 bytes.
/// Object keys use this format (without the preceding `S` marker).
fn read_string(data: &[u8], pos: &mut usize) -> Result<String, Error> {
    let len: usize = read_count(data, pos)?;
    if *pos + len > data.len() {
        return Err(Error::Format("UBJ string truncated".into()));
    }
    let s: String = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|_| Error::Format("UBJ string is not valid UTF-8".into()))?
        .to_owned();
    *pos += len;
    Ok(s)
}

/// Read any UBJSON integer type and return it as a `usize` (used for counts
/// and string lengths — must be non-negative).
fn read_count(data: &[u8], pos: &mut usize) -> Result<usize, Error> {
    let marker: u8 = read_byte(data, pos)?;
    let n: i64 = match marker {
        b'i' => read_i8(data, pos)? as i64,
        b'U' => read_u8(data, pos)? as i64,
        b'I' => read_i16(data, pos)? as i64,
        b'l' => read_i32(data, pos)? as i64,
        b'L' => read_i64(data, pos)?,
        m => {
            return Err(Error::Format(format!(
                "expected integer type for count, got {m:#04x}"
            )));
        }
    };
    if n < 0 {
        return Err(Error::Format(format!("negative UBJ count: {n}")));
    }
    Ok(n as usize)
}

// ---------------------------------------------------------------------------
// Value constructors
// ---------------------------------------------------------------------------

fn int_value(n: i64) -> Value {
    Value::Number(Number::from(n))
}

fn uint_value(n: u64) -> Value {
    Value::Number(Number::from(n))
}

fn float_value(f: f64) -> Result<Value, Error> {
    Number::from_f64(f)
        .map(Value::Number)
        .ok_or_else(|| Error::Format(format!("non-finite float {f} in UBJ")))
}

// ---------------------------------------------------------------------------
// Core parser
// ---------------------------------------------------------------------------

fn parse_value(data: &[u8], pos: &mut usize) -> Result<Value, Error> {
    let marker: u8 = read_byte(data, pos)?;
    match marker {
        b'N' => parse_value(data, pos), // no-op: skip and read next
        b'Z' => Ok(Value::Null),
        b'T' => Ok(Value::Bool(true)),
        b'F' => Ok(Value::Bool(false)),
        b'i' => Ok(int_value(read_i8(data, pos)? as i64)),
        b'U' => Ok(uint_value(read_u8(data, pos)? as u64)),
        b'I' => Ok(int_value(read_i16(data, pos)? as i64)),
        b'l' => Ok(int_value(read_i32(data, pos)? as i64)),
        b'L' => Ok(int_value(read_i64(data, pos)?)),
        b'd' => float_value(read_f32(data, pos)? as f64),
        b'D' => float_value(read_f64(data, pos)?),
        b'C' => Ok(Value::String((read_byte(data, pos)? as char).to_string())),
        b'S' => Ok(Value::String(read_string(data, pos)?)),
        b'H' => Ok(Value::String(read_string(data, pos)?)), // high-precision: keep as string
        b'[' => parse_array(data, pos),
        b'{' => parse_object(data, pos),
        m => Err(Error::Format(format!("unknown UBJ type marker {m:#04x}"))),
    }
}

/// Read a single value whose type is already known (used in typed containers).
fn read_typed(data: &[u8], pos: &mut usize, type_marker: u8) -> Result<Value, Error> {
    match type_marker {
        b'Z' => Ok(Value::Null),
        b'T' => Ok(Value::Bool(true)),
        b'F' => Ok(Value::Bool(false)),
        b'i' => Ok(int_value(read_i8(data, pos)? as i64)),
        b'U' => Ok(uint_value(read_u8(data, pos)? as u64)),
        b'I' => Ok(int_value(read_i16(data, pos)? as i64)),
        b'l' => Ok(int_value(read_i32(data, pos)? as i64)),
        b'L' => Ok(int_value(read_i64(data, pos)?)),
        b'd' => float_value(read_f32(data, pos)? as f64),
        b'D' => float_value(read_f64(data, pos)?),
        b'C' => Ok(Value::String((read_byte(data, pos)? as char).to_string())),
        b'S' => Ok(Value::String(read_string(data, pos)?)),
        m => Err(Error::Format(format!(
            "unsupported typed container element: {m:#04x}"
        ))),
    }
}

fn parse_array(data: &[u8], pos: &mut usize) -> Result<Value, Error> {
    if *pos >= data.len() {
        return Err(Error::Format("truncated UBJ array".into()));
    }

    // Optimized typed array: `[` `$` type `#` count  values...
    if data[*pos] == b'$' {
        *pos += 1;
        let type_marker = read_byte(data, pos)?;
        match read_byte(data, pos)? {
            b'#' => {}
            m => {
                return Err(Error::Format(format!(
                    "expected '#' after '$' in array, got {m:#04x}"
                )));
            }
        }
        let count = read_count(data, pos)?;
        let arr = (0..count)
            .map(|_| read_typed(data, pos, type_marker))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::Array(arr));
    }

    // Counted untyped array: `[` `#` count  values...
    if data[*pos] == b'#' {
        *pos += 1;
        let count: usize = read_count(data, pos)?;
        let arr: Vec<Value> = (0..count)
            .map(|_| parse_value(data, pos))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::Array(arr));
    }

    // Standard terminated array: `[` values... `]`
    let mut arr: Vec<Value> = Vec::new();
    loop {
        if *pos >= data.len() {
            return Err(Error::Format("unterminated UBJ array".into()));
        }
        if data[*pos] == b']' {
            *pos += 1;
            break;
        }
        arr.push(parse_value(data, pos)?);
    }
    Ok(Value::Array(arr))
}

fn parse_object(data: &[u8], pos: &mut usize) -> Result<Value, Error> {
    if *pos >= data.len() {
        return Err(Error::Format("truncated UBJ object".into()));
    }

    // Optimized typed object: `{` `$` type `#` count  (key value)...
    if data[*pos] == b'$' {
        *pos += 1;
        let type_marker: u8 = read_byte(data, pos)?;
        match read_byte(data, pos)? {
            b'#' => {}
            m => {
                return Err(Error::Format(format!(
                    "expected '#' after '$' in object, got {m:#04x}"
                )));
            }
        }
        let count: usize = read_count(data, pos)?;
        let mut map: Map<String, Value> = Map::with_capacity(count);
        for _ in 0..count {
            let key = read_string(data, pos)?;
            let value = read_typed(data, pos, type_marker)?;
            map.insert(key, value);
        }
        return Ok(Value::Object(map));
    }

    // Counted untyped object: `{` `#` count  (key value)...
    if data[*pos] == b'#' {
        *pos += 1;
        let count: usize = read_count(data, pos)?;
        let mut map: Map<String, Value> = Map::with_capacity(count);
        for _ in 0..count {
            let key = read_string(data, pos)?;
            let value = parse_value(data, pos)?;
            map.insert(key, value);
        }
        return Ok(Value::Object(map));
    }

    // Standard terminated object: `{` (key value)... `}`
    let mut map: Map<String, Value> = Map::new();
    loop {
        if *pos >= data.len() {
            return Err(Error::Format("unterminated UBJ object".into()));
        }
        if data[*pos] == b'}' {
            *pos += 1;
            break;
        }
        // Object keys are strings without the leading `S` marker.
        let key = read_string(data, pos)?;
        let value = parse_value(data, pos)?;
        map.insert(key, value);
    }
    Ok(Value::Object(map))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-encode a minimal UBJSON object and verify the parser round-trips it.
    #[test]
    fn parse_simple_object() {
        // Encodes: {"a": 1, "b": true}  (little-endian, terminated style)
        let data: &[u8] = &[
            b'{', // key "a" (length i8=1, then 'a')
            b'i', 1, b'a', // value: int8 42
            b'i', 42, // key "b"
            b'i', 1, b'b', // value: true
            b'T', b'}',
        ];
        let v: Value = parse(data).unwrap();
        assert_eq!(v["a"], Value::Number(42.into()));
        assert_eq!(v["b"], Value::Bool(true));
    }

    #[test]
    fn parse_typed_int32_array() {
        // Encodes: [1, -1, 2]  as an optimized i32 array (big-endian)
        let mut data: Vec<u8> = vec![b'[', b'$', b'l', b'#', b'i', 3];
        data.extend_from_slice(&1_i32.to_be_bytes());
        data.extend_from_slice(&(-1_i32).to_be_bytes());
        data.extend_from_slice(&2_i32.to_be_bytes());

        let v = parse(&data).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0], Value::Number(1.into()));
        assert_eq!(arr[1], Value::Number((-1_i64).into()));
        assert_eq!(arr[2], Value::Number(2.into()));
    }

    #[test]
    fn parse_typed_f32_array() {
        // Encodes: [0.5, -0.5]  as an optimized f32 array (big-endian)
        let mut data: Vec<u8> = vec![b'[', b'$', b'd', b'#', b'i', 2];
        data.extend_from_slice(&0.5_f32.to_be_bytes());
        data.extend_from_slice(&(-0.5_f32).to_be_bytes());

        let v: Value = parse(&data).unwrap();
        let arr: &Vec<Value> = v.as_array().unwrap();
        let a: f32 = arr[0].as_f64().unwrap() as f32;
        let b: f32 = arr[1].as_f64().unwrap() as f32;
        approx::assert_abs_diff_eq!(a, 0.5_f32, epsilon = 1e-7);
        approx::assert_abs_diff_eq!(b, -0.5_f32, epsilon = 1e-7);
    }

    #[test]
    fn parse_nested_string_values() {
        // Encodes: {"name": "hello"}
        let data: &[u8] = &[
            b'{', b'i', 4, b'n', b'a', b'm', b'e', b'S', b'i', 5, b'h', b'e', b'l', b'l', b'o',
            b'}',
        ];
        let v = parse(data).unwrap();
        assert_eq!(v["name"], Value::String("hello".into()));
    }
}
