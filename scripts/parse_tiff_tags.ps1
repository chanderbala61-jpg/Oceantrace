$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)
Write-Output "File Size: $($bytes.Length) bytes"

$isLittleEndian = ($bytes[0] -eq 0x49 -and $bytes[1] -eq 0x49)
$isBigEndian = ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x4D)
Write-Output "Endianness: $(if ($isBigEndian) { 'BigEndian (MM)' } else { 'LittleEndian (II)' })"

function Read-UInt16($offset) {
    if ($isBigEndian) {
        return ([uint16]$bytes[$offset] -shl 8) -bor [uint16]$bytes[$offset+1]
    } else {
        return [System.BitConverter]::ToUInt16($bytes, $offset)
    }
}

function Read-UInt32($offset) {
    if ($isBigEndian) {
        return ([uint32]$bytes[$offset] -shl 24) -bor ([uint32]$bytes[$offset+1] -shl 16) -bor ([uint32]$bytes[$offset+2] -shl 8) -bor [uint32]$bytes[$offset+3]
    } else {
        return [System.BitConverter]::ToUInt32($bytes, $offset)
    }
}

$firstIfdOffset = Read-UInt32 4
Write-Output "First IFD Offset: $firstIfdOffset"

$numEntries = Read-UInt16 $firstIfdOffset
Write-Output "Number of IFD entries: $numEntries"

$tagNames = @{
    256 = "ImageWidth"
    257 = "ImageLength"
    258 = "BitsPerSample"
    259 = "Compression"
    262 = "PhotometricInterpretation"
    273 = "StripOffsets"
    277 = "SamplesPerPixel"
    278 = "RowsPerStrip"
    279 = "StripByteCounts"
    284 = "PlanarConfiguration"
    339 = "SampleFormat"
    33550 = "ModelPixelScaleTag"
    33922 = "ModelTiepointTag"
    34735 = "GeoKeyDirectoryTag"
    34736 = "GeoDoubleParamsTag"
    34737 = "GeoAsciiParamsTag"
}

for ($i = 0; $i -lt $numEntries; $i++) {
    $entryOffset = $firstIfdOffset + 2 + ($i * 12)
    $tag = Read-UInt16 $entryOffset
    $type = Read-UInt16 ($entryOffset + 2)
    $count = Read-UInt32 ($entryOffset + 4)
    $valOffset = $entryOffset + 8
    
    $name = if ($tagNames.ContainsKey([int]$tag)) { $tagNames[[int]$tag] } else { "Tag_$tag" }
    
    $val = 0
    if ($count -eq 1 -and ($type -eq 3 -or $type -eq 1)) {
        $val = Read-UInt16 $valOffset
    } elseif ($count -eq 1 -and $type -eq 4) {
        $val = Read-UInt32 $valOffset
    } else {
        $val = "Offset/Pointer $(Read-UInt32 $valOffset) (count: $count, type: $type)"
    }
    
    Write-Output "Tag $tag ($name): $val"
}
