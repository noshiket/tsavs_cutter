#!/usr/bin/env python3
"""
MPEG-TS Cutter from AVS - AVSファイルのTrim指定に基づいてTSファイルをトリム

AVSファイルの例:
Trim(193,3188) ++ Trim(4988,22040) ++ Trim(23839,46345) ++ Trim(48145,48743)

使用方法:
python tsavs_cutter.py -i input.ts -a trim.avs -o output.ts
"""

from dataclasses import dataclass
from typing import List, Optional, Set
from collections import namedtuple
import argparse
import re
import sys


# ==============================================================================
# Constants
# ==============================================================================

TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47

PTS_CYCLE = 1 << 33  # 33-bit PTS/DTS cycle
PCR_CYCLE_27MHZ = 1 << 42  # 42-bit PCR cycle (27MHz)

# PSI Table IDs
PAT_TABLE_ID = 0x00
PMT_TABLE_ID = 0x02

# ARIB stream types
ARIB_VIDEO_STREAM_TYPES = {0x01, 0x02, 0x1B, 0x24}  # MPEG-1, MPEG-2, H.264, HEVC
ARIB_AUDIO_STREAM_TYPES = {0x03, 0x04, 0x0F, 0x11}  # MPEG-1 Audio, MPEG-2 Audio, AAC, LATM
ARIB_CAPTION_STREAM_TYPES = {0x06, 0x0D, 0x80, 0x81}  # Private data, DSM-CC type D, etc.

VideoFrame = namedtuple('VideoFrame', ['pts'])
TrimRange = namedtuple('TrimRange', ['start_frame', 'end_frame', 'start_pts', 'end_pts', 'offset_pts'])

CLOCK_FREQ = 90000  # 90 kHz for PTS/DTS


# ==============================================================================
# TSPacketUtil
# ==============================================================================

class TSPacketUtil:
    """MPEG-TS packet utility functions"""

    @staticmethod
    def get_pid(packet: bytes) -> int:
        """Extract PID from TS packet"""
        return ((packet[1] & 0x1F) << 8) | packet[2]

    @staticmethod
    def has_payload_start(packet: bytes) -> bool:
        """Check if packet has payload unit start indicator (PUSI)"""
        return (packet[1] & 0x40) != 0

    @staticmethod
    def has_adaptation_field(packet: bytes) -> bool:
        """Check if packet has adaptation field"""
        return (packet[3] & 0x20) != 0

    @staticmethod
    def has_pcr(packet: bytes) -> bool:
        """Check if packet contains PCR"""
        if not TSPacketUtil.has_adaptation_field(packet):
            return False
        if len(packet) < 6:
            return False
        af_len = packet[4]
        if af_len < 1:
            return False
        flags = packet[5]
        return (flags & 0x10) != 0

    @staticmethod
    def parse_pcr(packet: bytes) -> Optional[int]:
        """Parse PCR value from packet"""
        if not TSPacketUtil.has_pcr(packet):
            return None
        p = packet
        base = (p[6] << 25) | (p[7] << 17) | (p[8] << 9) | (p[9] << 1) | ((p[10] >> 7) & 0x01)
        ext = ((p[10] & 0x01) << 8) | p[11]
        return base * 300 + ext

    @staticmethod
    def write_pcr(packet: bytearray, new_pcr_val: int):
        """Write PCR value to packet"""
        if not TSPacketUtil.has_pcr(packet):
            return

        while new_pcr_val < 0:
            new_pcr_val += PCR_CYCLE_27MHZ
        new_pcr_val = new_pcr_val % PCR_CYCLE_27MHZ

        base = new_pcr_val // 300
        ext = new_pcr_val % 300

        packet[6] = (base >> 25) & 0xFF
        packet[7] = (base >> 17) & 0xFF
        packet[8] = (base >> 9) & 0xFF
        packet[9] = (base >> 1) & 0xFF
        packet[10] = ((base & 0x01) << 7) | 0x7E | ((ext >> 8) & 0x01)
        packet[11] = ext & 0xFF

    @staticmethod
    def get_pes_offset(packet: bytes) -> int:
        """Get PES payload offset in packet"""
        head_len = 4
        if TSPacketUtil.has_adaptation_field(packet):
            len_af = packet[4]
            head_len += 1 + len_af
        return head_len

    @staticmethod
    def parse_pts_dts(packet: bytes):
        """Parse PTS/DTS from PES header"""
        if not TSPacketUtil.has_payload_start(packet):
            return None, None

        offset = TSPacketUtil.get_pes_offset(packet)

        if offset + 9 > len(packet):
            return None, None

        if packet[offset] != 0x00 or packet[offset+1] != 0x00 or packet[offset+2] != 0x01:
            return None, None

        flags_2 = packet[offset + 7]
        pts_dts_flag = (flags_2 >> 6) & 0x03

        pes_header_data_length = packet[offset + 8]
        current_pos = offset + 9

        if pts_dts_flag == 0x00:
            return None, None

        if current_pos + pes_header_data_length > len(packet):
            return None, None

        def extract_pts_dts(p: bytes, pos: int) -> Optional[int]:
            if pos + 5 > len(p):
                return None
            return ((p[pos] & 0x0E) << 29) | \
                   ((p[pos+1] & 0xFF) << 22) | \
                   ((p[pos+2] & 0xFE) << 14) | \
                   ((p[pos+3] & 0xFF) << 7) | \
                   ((p[pos+4] & 0xFE) >> 1)

        pts = None
        dts = None

        if pts_dts_flag & 0x02:
            pts = extract_pts_dts(packet, current_pos)
            if pts is None:
                return None, None
            current_pos += 5

        if pts_dts_flag & 0x01:
            dts = extract_pts_dts(packet, current_pos)
            if dts is None and (pts_dts_flag == 0x03):
                return None, None

        return pts, dts

    @staticmethod
    def parse_pts(packet: bytes) -> Optional[int]:
        """Parse PTS from PES header"""
        pts, _ = TSPacketUtil.parse_pts_dts(packet)
        return pts

    @staticmethod
    def write_pts_dts(packet: bytearray, new_pts: Optional[int], new_dts: Optional[int]):
        """Write PTS/DTS to PES header"""
        if not TSPacketUtil.has_payload_start(packet):
            return

        offset = TSPacketUtil.get_pes_offset(packet)
        if offset + 9 > len(packet):
            return
        if packet[offset] != 0x00 or packet[offset+1] != 0x00 or packet[offset+2] != 0x01:
            return

        flags_2 = packet[offset + 7]
        pts_dts_flag = (flags_2 >> 6) & 0x03

        current_pos = offset + 9

        if pts_dts_flag == 0x00:
            return

        def write_33bit_val(p: bytearray, pos: int, val: int, marker_type: int):
            if pos + 5 > len(p):
                return
            while val < 0:
                val += PTS_CYCLE
            final_val = val % PTS_CYCLE

            marker = (marker_type << 4) | 0x01

            p[pos]   = marker | ((final_val >> 29) & 0x0E)
            p[pos+1] = (final_val >> 22) & 0xFF
            p[pos+2] = ((final_val >> 14) & 0xFE) | 0x01
            p[pos+3] = (final_val >> 7) & 0xFF
            p[pos+4] = ((final_val << 1) & 0xFE) | 0x01

        if (pts_dts_flag & 0x02) and new_pts is not None:
            write_33bit_val(packet, current_pos, new_pts, (pts_dts_flag >> 1) & 0x03)
            current_pos += 5

        if (pts_dts_flag & 0x01) and new_dts is not None:
            write_33bit_val(packet, current_pos, new_dts, 1)


# ==============================================================================
# StreamInfo
# ==============================================================================

@dataclass
class StreamInfo:
    """PAT/PMT解析結果を保持するデータクラス"""
    pmt_pid: int
    pcr_pid: int
    video_pid: int
    audio_pids: List[int]
    caption_pids: List[int]

    @property
    def elementary_pids(self) -> List[int]:
        """全elementary stream PIDのリスト（重複除去）"""
        pids = [self.video_pid] + self.audio_pids + self.caption_pids
        return list(set(pids))

    @property
    def psi_pids(self) -> List[int]:
        """PSI (PAT/PMT) PIDのリスト"""
        return [0x0000, self.pmt_pid]


# ==============================================================================
# ContinuityCounterManager
# ==============================================================================

class ContinuityCounterManager:
    """PIDごとのContinuity Counter管理"""

    def __init__(self):
        self.cc_state = {}

    def get_next_cc(self, pid: int) -> int:
        """次のCCを取得（0-15の循環）"""
        if pid not in self.cc_state:
            self.cc_state[pid] = 0
        else:
            self.cc_state[pid] = (self.cc_state[pid] + 1) & 0x0F
        return self.cc_state[pid]


# ==============================================================================
# StreamAnalyzer
# ==============================================================================

class StreamAnalyzer:
    """ストリーム解析とインデックス構築"""

    def analyze_stream(self, input_file: str) -> StreamInfo:
        """Analyze PAT/PMT and return StreamInfo"""
        pmt_pid: int = -1
        pcr_pid: int = -1
        video_pid: int = -1
        audio_pids: List[int] = []
        caption_pids: List[int] = []
        elementary_pids: Set[int] = set()

        with open(input_file, 'rb') as f:
            # First pass: Find PMT PID from PAT
            chunk_size = TS_PACKET_SIZE * 5000
            data = f.read(chunk_size)

            for i in range(0, len(data), TS_PACKET_SIZE):
                pkt = data[i:i+TS_PACKET_SIZE]
                if len(pkt) < TS_PACKET_SIZE:
                    continue

                if TSPacketUtil.get_pid(pkt) == 0x0000 and TSPacketUtil.has_payload_start(pkt):
                    offset = TSPacketUtil.get_pes_offset(pkt)
                    if offset >= len(pkt):
                        continue
                    offset += 1 + pkt[offset]  # pointer_field
                    if offset + 8 >= len(pkt):
                        continue
                    if pkt[offset] != PAT_TABLE_ID:
                        continue
                    offset += 8  # Skip to program loop

                    while offset + 4 <= len(pkt):
                        prog_num = (pkt[offset] << 8) | pkt[offset+1]
                        if prog_num != 0:
                            pmt_pid = ((pkt[offset+2] & 0x1F) << 8) | pkt[offset+3]
                            break
                        offset += 4
                    if pmt_pid != -1:
                        break

            if pmt_pid == -1:
                pmt_pid = 0x1000

            # Second pass: Scan ALL PMTs
            f.seek(0)
            while True:
                data = f.read(chunk_size)
                if not data:
                    break

                for i in range(0, len(data), TS_PACKET_SIZE):
                    pkt = data[i:i+TS_PACKET_SIZE]
                    if len(pkt) < TS_PACKET_SIZE:
                        continue

                    if TSPacketUtil.get_pid(pkt) == pmt_pid and TSPacketUtil.has_payload_start(pkt):
                        offset = TSPacketUtil.get_pes_offset(pkt)
                        if offset >= len(pkt):
                            continue
                        offset += 1 + pkt[offset]  # pointer_field
                        if offset + 12 >= len(pkt):
                            continue
                        if pkt[offset] != PMT_TABLE_ID:
                            continue
                        offset += 8
                        pcr_pid = ((pkt[offset] & 0x1F) << 8) | pkt[offset+1]
                        offset += 2
                        program_info_len = ((pkt[offset] & 0x0F) << 8) | pkt[offset+1]
                        offset += 2
                        stream_offset = offset + program_info_len

                        while stream_offset + 5 < len(pkt) - 4:
                            s_type = pkt[stream_offset]
                            pid = ((pkt[stream_offset+1] & 0x1F) << 8) | pkt[stream_offset+2]
                            es_len = ((pkt[stream_offset+3] & 0x0F) << 8) | pkt[stream_offset+4]
                            elementary_pids.add(pid)

                            if stream_offset + 5 + es_len > len(pkt) - 4:
                                break

                            if video_pid == -1 and s_type in ARIB_VIDEO_STREAM_TYPES:
                                video_pid = pid
                            elif s_type in ARIB_AUDIO_STREAM_TYPES:
                                if pid not in audio_pids:
                                    audio_pids.append(pid)
                            elif s_type in ARIB_CAPTION_STREAM_TYPES:
                                if pid not in caption_pids:
                                    caption_pids.append(pid)
                            stream_offset += 5 + es_len

        if video_pid == -1:
            if elementary_pids:
                video_pid = list(elementary_pids)[0]
            else:
                raise Exception("Video stream not found.")

        return StreamInfo(
            pmt_pid=pmt_pid,
            pcr_pid=pcr_pid,
            video_pid=video_pid,
            audio_pids=audio_pids,
            caption_pids=caption_pids
        )

    def build_video_index(self, input_file: str, video_pid: int) -> List[VideoFrame]:
        """Build video frame index"""
        video_index: List[VideoFrame] = []
        chunk_size: int = TS_PACKET_SIZE * 10000

        with open(input_file, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                n_packets: int = len(chunk) // TS_PACKET_SIZE
                for i in range(n_packets):
                    offset: int = i * TS_PACKET_SIZE
                    packet_bytes = chunk[offset:offset+TS_PACKET_SIZE]

                    if len(packet_bytes) < TS_PACKET_SIZE:
                        break
                    if packet_bytes[0] != SYNC_BYTE:
                        continue

                    if TSPacketUtil.get_pid(packet_bytes) == video_pid:
                        if TSPacketUtil.has_payload_start(packet_bytes):
                            pts, _ = TSPacketUtil.parse_pts_dts(packet_bytes)
                            if pts is not None:
                                video_index.append(VideoFrame(pts))

        return video_index


def calculate_trim_ranges(trim_specs: List[tuple], video_index: List[VideoFrame]) -> List[TrimRange]:
    """
    trim指定から抽出範囲を計算（offset_pts付き）

    Args:
        trim_specs: trim指定リスト（例: [(0, 200), (250, 340)]）
        video_index: ビデオフレームインデックス

    Returns:
        List[TrimRange]: 抽出範囲リスト（offset_pts計算済み）
    """
    if not video_index:
        raise ValueError("No video frames found in input file")

    total_frames = len(video_index)
    result = []
    cumulative_end_pts = 0

    for i, (start_frame, end_frame) in enumerate(trim_specs):
        # 範囲チェック
        if start_frame < 0:
            raise ValueError(f"Frame {start_frame} is out of range (must be >= 0)")
        if end_frame >= total_frames:
            raise ValueError(f"Frame {end_frame} is out of range (total frames: {total_frames})")
        if start_frame > end_frame:
            raise ValueError(f"Invalid range [{start_frame},{end_frame}]: start must be <= end")

        # フレーム番号からPTSを取得
        start_pts = video_index[start_frame].pts
        # end_frameは含まれるので、次のフレーム（end_frame + 1）の開始時刻を取得
        if end_frame + 1 < total_frames:
            end_pts = video_index[end_frame + 1].pts
        else:
            # 最終フレームの場合、end_frameのPTSを使用
            end_pts = video_index[end_frame].pts

        # オフセット計算
        offset_pts = 0 if i == 0 else cumulative_end_pts - start_pts

        result.append(TrimRange(
            start_frame=start_frame,
            end_frame=end_frame,
            start_pts=start_pts,
            end_pts=end_pts,
            offset_pts=offset_pts
        ))

        cumulative_end_pts = (end_pts + offset_pts + PTS_CYCLE) % PTS_CYCLE

    return result


def parse_avs_file(avs_file: str) -> List[tuple]:
    """
    AVSファイルを読み込んでTrim範囲を抽出

    Args:
        avs_file: AVSファイルパス

    Returns:
        List[tuple]: Trim範囲のリスト（例: [(193, 3188), (4988, 22040)]）
    """
    with open(avs_file, 'r', encoding='utf-8') as f:
        content = f.read()

    trim_pattern = re.compile(r'Trim\((\d+),(\d+)\)')
    matches = trim_pattern.findall(content)
    return [(int(start), int(end)) for start, end in matches]


# ==============================================================================
# Main Trimmer
# ==============================================================================

def trim_and_write_output(input_file: str, output_file: str, trim_specs: List[tuple]) -> None:
    """
    Trim mode: Extract frame ranges from input TS file

    Args:
        input_file: Input TS file path
        output_file: Output TS file path
        trim_specs: Trim specifications (e.g., [(0, 200), (250, 340)])
    """
    print(f"Input file: {input_file}")
    print(f"Total video frames: (analyzing...)")
    print()

    # 1. Stream analysis
    print("Analyzing streams...")
    analyzer = StreamAnalyzer()
    stream_info = analyzer.analyze_stream(input_file)
    print(f"  Video PID: 0x{stream_info.video_pid:x}, Audio PID: {[hex(p) for p in stream_info.audio_pids]}, Caption PID: {[hex(p) for p in stream_info.caption_pids]}")
    print()

    print("Building video index...")
    video_index = analyzer.build_video_index(input_file, stream_info.video_pid)
    total_frames: int = len(video_index)
    print(f"Found {total_frames} video frames (PTS range: {video_index[0].pts/CLOCK_FREQ:.3f} - {video_index[-1].pts/CLOCK_FREQ:.3f})")
    print()

    # 2. Calculate trim ranges
    trim_ranges = calculate_trim_ranges(trim_specs, video_index)

    # 3. Process and concatenate segments
    print("Processing trim segments...")
    total_output_packets: int = 0
    total_output_frames: int = 0
    cc_manager = ContinuityCounterManager()

    with open(output_file, 'wb') as out_f:
        for seg_idx, trim_range in enumerate(trim_ranges):
            print(f"\nTrim segment {seg_idx + 1}: frames [{trim_range.start_frame}, {trim_range.end_frame}]")
            duration_sec = (trim_range.end_pts - trim_range.start_pts + PTS_CYCLE) % PTS_CYCLE / CLOCK_FREQ
            print(f"  Source PTS: {trim_range.start_pts/CLOCK_FREQ:.3f} - {trim_range.end_pts/CLOCK_FREQ:.3f} (duration: {duration_sec:.3f}s, {trim_range.end_frame - trim_range.start_frame + 1} frames)")
            print(f"  Output PTS: {(trim_range.start_pts + trim_range.offset_pts + PTS_CYCLE) % PTS_CYCLE / CLOCK_FREQ:.3f} - {(trim_range.end_pts + trim_range.offset_pts + PTS_CYCLE) % PTS_CYCLE / CLOCK_FREQ:.3f} (offset: {trim_range.offset_pts/CLOCK_FREQ:+.3f}s)")

            in_range: bool = False
            video_count: int = 0
            audio_count: int = 0
            caption_count: int = 0

            with open(input_file, 'rb') as in_f:
                while True:
                    packet = in_f.read(TS_PACKET_SIZE)
                    if len(packet) < TS_PACKET_SIZE:
                        break

                    pid = TSPacketUtil.get_pid(packet)

                    # Get PTS if present
                    packet_pts: Optional[int] = None
                    if TSPacketUtil.has_payload_start(packet):
                        packet_pts = TSPacketUtil.parse_pts(packet)

                    # Check PTS range
                    if packet_pts is not None:
                        if packet_pts < trim_range.start_pts:
                            continue
                        elif packet_pts <= trim_range.end_pts:
                            in_range = True
                        else:
                            if in_range:
                                break
                            else:
                                continue

                    # Include PAT/PMT/PCR packets (packets without PTS)
                    if (not in_range) and (packet_pts is None):
                        if pid not in [0x00, stream_info.pmt_pid]:
                            continue

                    # Copy packet and adjust timestamps
                    packet_copy = bytearray(packet)

                    # Adjust PCR
                    if TSPacketUtil.has_pcr(packet_copy):
                        pcr_val = TSPacketUtil.parse_pcr(packet_copy)
                        if pcr_val is not None:
                            offset_pcr: int = int(trim_range.offset_pts * 300)
                            new_pcr: int = (pcr_val + offset_pcr + PCR_CYCLE_27MHZ) % PCR_CYCLE_27MHZ
                            TSPacketUtil.write_pcr(packet_copy, new_pcr)

                    # Adjust PTS/DTS
                    if TSPacketUtil.has_payload_start(packet_copy):
                        pts_val, dts_val = TSPacketUtil.parse_pts_dts(packet_copy)

                        new_pts: Optional[int] = (pts_val + trim_range.offset_pts + PTS_CYCLE) % PTS_CYCLE if pts_val is not None else None
                        new_dts: Optional[int] = (dts_val + trim_range.offset_pts + PTS_CYCLE) % PTS_CYCLE if dts_val is not None else None

                        if (new_pts is not None) or (new_dts is not None):
                            TSPacketUtil.write_pts_dts(packet_copy, new_pts, new_dts)

                    # Update Continuity Counter
                    has_payload: bool = (packet_copy[3] & 0x10) != 0
                    if has_payload:
                        new_cc: int = cc_manager.get_next_cc(pid)
                        packet_copy[3] = (packet_copy[3] & 0xF0) | new_cc

                    # Write output
                    out_f.write(packet_copy)
                    total_output_packets += 1

                    # Count packets by type
                    if pid == stream_info.video_pid:
                        video_count += 1
                        if TSPacketUtil.has_payload_start(packet):
                            total_output_frames += 1
                    elif pid in stream_info.audio_pids:
                        audio_count += 1
                    elif pid in stream_info.caption_pids:
                        caption_count += 1

            print(f"  Packets: Video={video_count}, Audio={audio_count}, Caption={caption_count}")

    final_pts = (trim_ranges[-1].end_pts + trim_ranges[-1].offset_pts + PTS_CYCLE) % PTS_CYCLE if trim_ranges else 0
    print(f"\nTotal output: {final_pts/CLOCK_FREQ:.3f}s, {total_output_frames} frames, {total_output_packets} packets")
    print(f"Written: {output_file} ({total_output_packets * TS_PACKET_SIZE / 1024 / 1024:.1f} MB)")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MPEG-TS Cutter from AVS - Extract frame ranges specified in AVS file"
    )
    parser.add_argument("-i", "--input", required=True, help="Input TS file")
    parser.add_argument("-a", "--avs", required=True, help="AVS file with Trim() specifications")
    parser.add_argument("-o", "--output", required=True, help="Output TS file")

    args = parser.parse_args()

    # AVSファイルをパース
    print(f"Parsing AVS file: {args.avs}")
    try:
        trim_specs = parse_avs_file(args.avs)
    except Exception as e:
        print(f"Error parsing AVS file: {e}", file=sys.stderr)
        sys.exit(1)

    if not trim_specs:
        print("Error: No Trim() specifications found in AVS file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trim_specs)} trim ranges:")
    for i, (start, end) in enumerate(trim_specs, 1):
        print(f"  {i}. [{start},{end}]")
    print()

    # トリム処理
    try:
        trim_and_write_output(args.input, args.output, trim_specs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
