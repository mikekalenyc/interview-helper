import json
from pathlib import Path

import pytest

from interview_helper.library import TechnicalLibrary


def library_files(root: Path) -> Path:
    snapshot = root / 'snapshots' / 'one'
    section = snapshot / 'aws-tgw' / 'sections'
    section.mkdir(parents=True)
    (root / 'LATEST.txt').write_text('snapshots/one')
    (section / 'pages.md').write_text('# Transit Gateway\n## PDF page 10\nAppliance mode\nTransit Gateway appliance mode maintains flow symmetry through the same availability zone for stateful network inspection.\n## PDF page 11\nTransit Gateway route tables select attachment routes and support static routing and propagation.\n')
    (snapshot / 'aws-tgw' / 'original.txt').write_text('UNINDEXEDORIGINAL secret text')
    cf = snapshot / 'cloudflare'
    cf.mkdir()
    (cf / 'reference.md').write_text('# DNS\n## Records\nCloudflare DNS records provide authoritative domain resolution and record configuration.\n')
    (snapshot / 'manifest.json').write_text(json.dumps({'documents': [
        dict(title='Transit Gateway Guide', source_url='https://docs.aws.amazon.com/tgw.pdf', downloaded_at='2026-09-07', directory='snapshots/one/aws-tgw', vendor='AWS', status='downloaded'),
        dict(title='Cloudflare DNS', source_url='https://developers.cloudflare.com/dns/', downloaded_at='2026-09-07', directory='snapshots/one/cloudflare', vendor='Cloudflare', status='downloaded')]}))
    return snapshot


def test_relevance_provenance_aliases_and_single_representation(tmp_path: Path) -> None:
    library_files(tmp_path)
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    results = library.search('Why does TGW appliance mode keep symmetric flows?')
    assert 'flow symmetry' in results[0].text
    assert results[0].title == 'Transit Gateway Guide'
    assert results[0].url == 'https://docs.aws.amazon.com/tgw.pdf'
    assert results[0].downloaded_at == '2026-09-07'
    assert results[0].locator == 'PDF physical page 10'
    assert library.search('UNINDEXEDORIGINAL') == ()
    library.close()


def test_refresh_and_reuse(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    cache = tmp_path / 'cache.sqlite'
    for _ in range(2):
        library = TechnicalLibrary.open(tmp_path, cache_path=cache)
        assert library.search('appliance')
        library.close()
    (snapshot / 'cloudflare' / 'reference.md').write_text('# Refreshed\nQuicksilver is a new reference passage with enough content to be indexed as local technical documentation.\n')
    library = TechnicalLibrary.open(tmp_path, cache_path=cache)
    assert library.search('quicksilver')
    library.close()


def test_question_phrases_beat_repeated_generic_routing_words(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    section = snapshot / 'aws-tgw' / 'sections' / 'lookup.md'
    section.write_text(
        '# Subnet types\nPrivate subnet – The subnet does not have a direct route '
        'to an internet gateway. Resources in a private subnet can use a NAT device '
        'for outbound internet access.\n'
        '# Multiple VPC peering connections\nVPC peering is not transitive. '
        'If VPC A peers with VPC B and VPC B peers with VPC C, VPC A cannot '
        'use VPC B as a transit point to VPC C; A and C need a direct peering.\n'
        '# Interface setup\n' + ('Traffic through a connection from a workload '
        'can reach another network. Need your own connection to carry traffic.\n' * 8)
        + '# Private subnet CLI procedure\n'
        + ('aws ec2 create-route --private-subnet subnet-id --internet gateway\n' * 8)
    )
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    try:
        subnet = library.search(
            'Does placing a workload in a private subnet stop it from reaching the internet?',
            max_characters=2000, limit=2,
        )
        assert 'does not have a direct route' in subnet[0].text
        peering = library.search(
            'Can VPC peering carry traffic from VPC A through VPC B to VPC C, '
            'or do A and C need their own connection?', max_characters=2000, limit=2,
        )
        assert 'not transitive' in peering[0].text
        assert sum(len(r.text) for r in peering) <= 2000
    finally:
        library.close()


def test_bounds_and_match_injection(tmp_path: Path) -> None:
    library_files(tmp_path)
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    assert library.search('" OR * : NEAR() NOT "') == ()
    assert library.search('appliance', max_characters=0) == ()
    results = library.search('transit gateway', max_characters=55, limit=1)
    assert len(results) == 1
    assert sum(len(r.text) for r in results) <= 55
    library.close()


def test_manifest_escape_and_missing_text_are_actionable(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    manifest_path = snapshot / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['documents'][0]['directory'] = '../../outside'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes its root'):
        TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    (tmp_path / 'LATEST.txt').write_text('/etc')
    with pytest.raises(ValueError, match='escapes its root'):
        TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')


def test_heading_and_toc_filtering(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    (snapshot / 'cloudflare' / 'reference.md').write_text('# Tunnel\n## Failover\nFailover........................... 22\nCloudflare tunnel replicas improve availability when connectors fail and serve private network traffic.\n')
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    result = library.search('tunnel failover')[0]
    assert result.locator == 'Failover'
    assert '.....' not in result.text
    library.close()


def test_snapshot_refresh_and_symlink_safety(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    cache = tmp_path / 'cache.sqlite'
    library = TechnicalLibrary.open(tmp_path, cache_path=cache)
    library.close()
    import shutil
    newer = tmp_path / 'snapshots' / 'two'
    shutil.copytree(snapshot, newer)
    manifest_file = newer / 'manifest.json'
    manifest_file.write_text(manifest_file.read_text().replace('snapshots/one', 'snapshots/two').replace('2026-09-07', '2026-09-08'))
    (tmp_path / 'LATEST.txt').write_text('snapshots/two')
    library = TechnicalLibrary.open(tmp_path, cache_path=cache)
    assert library.search('appliance')[0].downloaded_at == '2026-09-08'
    library.close()
    target = newer / 'cloudflare' / 'reference.md'
    target.unlink()
    target.symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='escapes its root'):
        TechnicalLibrary.open(tmp_path, cache_path=cache)


def test_oversized_reference_rejected(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    reference = snapshot / 'cloudflare' / 'reference.md'
    with reference.open('wb') as stream:
        stream.truncate(4_000_001)
    with pytest.raises(ValueError, match='indexing limits'):
        TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')


def test_generic_followup_reuses_topic_but_explicit_switch_does_not(tmp_path: Path) -> None:
    library_files(tmp_path)
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    prior = 'Why is TGW appliance mode needed?'
    for question in ('How does that work?', 'What are the tradeoffs?', 'How does it handle failures?'):
        result = library.search(question, recent_question=prior)
        assert result
        assert result[0].title == 'Transit Gateway Guide'
        assert 'appliance' in result[0].text.lower()
    switched = library.search('What about DNS?', recent_question=prior)
    assert switched
    assert all(result.title == 'Cloudflare DNS' for result in switched)
    assert library.search('', recent_question=prior) == ()
    assert library.search('How does that work?') == ()
    library.close()


def test_small_budget_keeps_matching_text_and_limitations_followup(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    (snapshot / 'cloudflare' / 'reference.md').write_text(
        '# Security\n' + 'General introductory context with no specific controls.\n' * 20
        + 'Network ACLs are stateless and require explicit return traffic rules.\n'
    )
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    result = library.search('What are its limitations?', recent_question='Explain network ACLs', max_characters=180, limit=1)
    assert result and 'stateless' in result[0].text
    assert len(result[0].text) <= 180
    library.close()


def test_direct_subject_and_followup_intent_beat_incidental_mentions(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    (snapshot / 'aws-tgw' / 'sections' / 'pages.md').write_text(
        '# Networking\n## PDF page 1\nCompare security groups and network ACLs\n'
        'Security groups are stateful. Network ACLs are stateless.\n'
        '## PDF page 2\nNetwork ACL limitations\n'
        'Network ACLs do not evaluate traffic within a subnet. Security groups still apply to resources.\n'
        '## PDF page 3\nNetwork analyzer\n'
        'The analyzer checks a security group, network ACL, route table and interface for network problems.\n'
    )
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    prior = 'Compare security groups and NACLs'
    assert library.search(prior, limit=1)[0].locator == 'PDF physical page 1'
    result = library.search('What are its limitations?', recent_question=prior, limit=1)
    assert result[0].locator == 'PDF physical page 2'
    assert 'within a subnet' in result[0].text
    library.close()


def test_numeric_table_values_survive_indexing(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    (snapshot / 'aws-tgw' / 'sections' / 'pages.md').write_text(
        '# Quotas\n## PDF page 1\nNetwork ACL quotas\nRules per network ACL\n20\n'
        'The default rule quota applies independently to inbound and outbound rules.\n'
    )
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    assert '\n20\n' in library.search('NACL quotas')[0].text
    library.close()


def test_multi_control_limitations_include_each_controls_evidence(tmp_path: Path) -> None:
    snapshot = library_files(tmp_path)
    (snapshot / 'aws-tgw' / 'sections' / 'pages.md').write_text(
        '# Networking\n## PDF page 1\nSecurity group quotas\n'
        'Security groups allow 60 inbound and 60 outbound rules by default.\n'
        '## PDF page 2\nNetwork ACL quotas\n'
        'Network ACLs allow 20 inbound and 20 outbound rules by default.\n'
        '## PDF page 3\nNetwork ACL limitations\n'
        'Network ACLs do not evaluate traffic within a subnet.\n'
    )
    library = TechnicalLibrary.open(tmp_path, cache_path=tmp_path / 'cache.sqlite')
    result = library.search('What are their limitations?', recent_question='Compare security groups and NACLs', limit=2, max_characters=500)
    assert len(result) == 2
    assert '60 inbound' in result[0].text
    assert '20 inbound' in result[1].text or 'within a subnet' in result[1].text
    assert sum(len(r.text) for r in result) <= 500
    library.close()
