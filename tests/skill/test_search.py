# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

from ms_agent.skill.search import SkillSearchEngine


class _Catalog:

    def __init__(self):
        self._cache_version = 0
        self.skills = {
            'alpha': SimpleNamespace(
                name='Alpha',
                description='alpha searchable skill',
                tags=['demo'],
                content='alpha content',
            )
        }

    def get_enabled_skills(self):
        return self.skills


def test_search_clears_index_when_all_skills_removed():
    catalog = _Catalog()
    engine = SkillSearchEngine(catalog, backend='bm25')

    assert engine.search('alpha')[0][0] == 'alpha'

    catalog.skills = {}
    catalog._cache_version += 1

    assert engine.search('alpha') == []


def test_search_reindexes_after_empty_catalog_is_repopulated():
    catalog = _Catalog()
    engine = SkillSearchEngine(catalog, backend='bm25')
    assert engine.search('alpha')

    catalog.skills = {}
    catalog._cache_version += 1
    assert engine.search('alpha') == []

    catalog.skills = {
        'beta': SimpleNamespace(
            name='Beta',
            description='beta searchable skill',
            tags=['demo'],
            content='beta content',
        )
    }
    catalog._cache_version += 1

    results = engine.search('beta')
    assert results and results[0][0] == 'beta'
