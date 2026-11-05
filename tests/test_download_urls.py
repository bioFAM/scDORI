"""Default reference downloads must come from a host whose certificate matches its name."""

import stat

import pytest

from scdori.pp.download import download_file, resolve_genome_urls


@pytest.mark.parametrize(("species", "assembly"), [("mouse", "mm10"), ("human", "hg38")])
def test_default_ucsc_urls_use_the_host_named_on_its_certificate(species, assembly):
    _, chrom_sizes, fasta = resolve_genome_urls(species, assembly, None, None, None)
    for url in (chrom_sizes, fasta):
        assert url.startswith("https://hgdownload.soe.ucsc.edu/goldenPath/")


def test_urls_you_supply_are_used_unchanged():
    assert resolve_genome_urls("mouse", "mm10", "a", "b", "c") == ("a", "b", "c")


@pytest.fixture
def fake_wget(tmp_path, monkeypatch):
    """A stand-in wget that, like the real one, creates its output file before it can fail."""
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "wget"
    script.write_text('#!/bin/sh\nprintf partial > "$2"\n[ -z "$WGET_FAILS" ] || exit 4\nprintf "%s" "$3" > "$2"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{folder}:/usr/bin:/bin")
    return tmp_path


def test_failed_download_leaves_nothing_that_looks_finished(fake_wget, monkeypatch):
    target = fake_wget / "mm10.chrom.sizes"
    monkeypatch.setenv("WGET_FAILS", "1")
    with pytest.raises(RuntimeError, match="Could not download"):
        download_file("https://example.org/file", target)
    assert list(fake_wget.glob("mm10.chrom.sizes*")) == []
    # The next attempt downloads again instead of accepting a leftover file.
    monkeypatch.delenv("WGET_FAILS")
    download_file("https://example.org/file", target)
    assert target.read_text() == "https://example.org/file"


def test_address_with_shell_characters_reaches_wget_unchanged(fake_wget):
    target = fake_wget / "a b.gtf"
    address = "https://example.org/file?a=1&b=2"
    download_file(address, target)
    assert target.read_text() == address


def test_finished_file_is_not_downloaded_again(fake_wget):
    target = fake_wget / "done.txt"
    target.write_text("kept")
    download_file("https://example.org/file", target)
    assert target.read_text() == "kept"

