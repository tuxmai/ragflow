#
#  Copyright 2024 The Tuxmai Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Qdrant imports - these would need to be installed: pip install qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import CountRequest, Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, Record, ScoredPoint, ScrollRequest, SearchRequest, UpdateStatus, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    # Mock classes for type checking when qdrant-client is not installed
    QDRANT_AVAILABLE = False
    logging.warning("qdrant-client not installed. Install with: pip install qdrant-client")

    # Create mock classes to prevent import errors
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("qdrant-client not installed")

    class Filter:
        def __init__(self, *args, **kwargs):
            pass

    class FieldCondition:
        def __init__(self, *args, **kwargs):
            pass

    class MatchValue:
        def __init__(self, *args, **kwargs):
            pass

    class MatchAny:
        def __init__(self, *args, **kwargs):
            pass

    class ScoredPoint:
        def __init__(self):
            self.id = None
            self.payload = {}
            self.vector = None

    class Record:
        def __init__(self):
            self.id = None
            self.payload = {}


from rag.nlp import query, rag_tokenizer
from rag.settings import PAGERANK_FLD, TAG_FLD
from rag.utils import get_float, rmSpace


def index_name(uid):
    return f"ragflow_{uid}"


class QdrantDealer:
    def __init__(self, client: QdrantClient):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")
        self.qryr = query.FulltextQueryer()
        self.client = client

    @dataclass
    class SearchResult:
        total: int
        ids: List[str]
        query_vector: Optional[List[float]] = None
        field: Optional[Dict[str, Dict[str, Any]]] = None
        highlight: Optional[Dict[str, str]] = None
        aggregation: Optional[Union[List, Dict]] = None
        keywords: Optional[List[str]] = None
        group_docs: Optional[List[List]] = None

    def get_vector(self, txt: str, emb_mdl, topk: int = 10, similarity: float = 0.1):
        """Generate vector embedding for search query"""
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(f"QdrantDealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        return embedding_data, topk, similarity

    def get_filters(self, req: Dict[str, Any]) -> Optional[Filter]:
        """Convert request parameters to Qdrant filters"""
        if not QDRANT_AVAILABLE:
            return None

        conditions = []

        # Handle kb_ids and doc_ids
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                if isinstance(req[key], list):
                    conditions.append(FieldCondition(key=field, match=MatchAny(any=req[key])))
                else:
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=req[key])))

        # Handle other fields
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                if isinstance(req[key], list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=req[key])))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=req[key])))

        return Filter(must=conditions) if conditions else None

    def _build_text_query_filter(self, match_text_expr):
        """Build text search filter from MatchTextExpr"""
        # This is a simplified implementation - in practice, you'd need to
        # implement full-text search capabilities or use a hybrid approach
        # For now, we'll return None and rely on vector search
        return None

    def search(
        self, req: Dict[str, Any], idx_names: Union[str, List[str]], kb_ids: List[str], emb_mdl=None, highlight: bool = False, rank_feature: Optional[Dict] = None
    ) -> "QdrantDealer.SearchResult":
        """Main search method compatible with original Dealer interface"""
        if not QDRANT_AVAILABLE:
            return self.SearchResult(total=0, ids=[], field={})

        filters = self.get_filters(req)

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get(
            "fields",
            [
                "docnm_kwd",
                "content_ltks",
                "kb_id",
                "img_id",
                "title_tks",
                "important_kwd",
                "position_int",
                "doc_id",
                "page_num_int",
                "top_int",
                "create_timestamp_flt",
                "knowledge_graph_kwd",
                "question_kwd",
                "question_tks",
                "doc_type_kwd",
                "available_int",
                "content_with_weight",
                PAGERANK_FLD,
                TAG_FLD,
            ],
        )

        kwds = set()
        qst = req.get("question", "")
        q_vec: List[float] = []

        # Handle collection names
        if isinstance(idx_names, str):
            collection_names = [idx_names]
        else:
            collection_names = idx_names

        if not qst:
            # No question - just retrieve documents with filters
            results = []
            total = 0

            for collection_name in collection_names:
                try:
                    # Use scroll to get documents with filters
                    scroll_result = self.client.scroll(collection_name=collection_name, scroll_filter=filters, limit=limit, offset=offset, with_payload=True, with_vectors=False)
                    results.extend(scroll_result[0])  # scroll returns (points, next_page_offset)

                    # Get total count
                    count_result = self.client.count(collection_name=collection_name, count_filter=filters)
                    total += count_result.count

                except Exception as e:
                    logging.warning(f"Error searching collection {collection_name}: {e}")
                    continue

            logging.debug("QdrantDealer.search TOTAL: {}".format(total))

        else:
            # Question-based search
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)

            if emb_mdl is None:
                # Text-only search - limited support in Qdrant
                # This would require implementing full-text search capabilities
                results = []
                total = 0
                logging.warning("Text-only search not fully implemented for Qdrant")

            else:
                # Vector + text hybrid search
                q_vec, vector_topk, similarity_threshold = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))

                results = []
                total = 0

                for collection_name in collection_names:
                    try:
                        # Perform vector search
                        search_result = self.client.search(
                            collection_name=collection_name,
                            query_vector=q_vec,
                            query_filter=filters,
                            limit=limit,
                            offset=offset,
                            score_threshold=similarity_threshold,
                            with_payload=True,
                            with_vectors=True,
                        )
                        results.extend(search_result)

                        # Get approximate total (Qdrant doesn't provide exact total for vector search)
                        total += len(search_result)

                    except Exception as e:
                        logging.warning(f"Error searching collection {collection_name}: {e}")
                        continue

                # If no results, try with lower similarity threshold
                if total == 0 and not filters:
                    for collection_name in collection_names:
                        try:
                            search_result = self.client.search(
                                collection_name=collection_name, query_vector=q_vec, limit=limit, offset=offset, score_threshold=0.17, with_payload=True, with_vectors=True
                            )
                            results.extend(search_result)
                            total += len(search_result)

                        except Exception as e:
                            logging.warning(f"Error in fallback search for collection {collection_name}: {e}")
                            continue

                logging.debug("QdrantDealer.search TOTAL: {}".format(total))

            # Process keywords
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        # Process results
        ids: List[str] = []
        fields: Dict[str, Dict[str, Any]] = {}
        highlights: Dict[str, str] = {}

        for result in results:
            if isinstance(result, ScoredPoint):
                point_id = str(result.id)
                ids.append(point_id)
                fields[point_id] = result.payload or {}
                # Add vector to payload if available
                if result.vector:
                    vector_column_name = f"q_{len(q_vec)}_vec" if q_vec else "vector"
                    fields[point_id][vector_column_name] = result.vector
            elif isinstance(result, Record):
                point_id = str(result.id)
                ids.append(point_id)
                fields[point_id] = result.payload or {}

        # Simple highlight implementation (would need more sophisticated implementation)
        if highlight and qst:
            for point_id in ids:
                content = fields[point_id].get("content_with_weight", "")
                highlights[point_id] = content  # Simplified - no actual highlighting

        # Simple aggregation (document names)
        aggregation: Dict[str, int] = {}
        for point_id in ids:
            doc_name = fields[point_id].get("docnm_kwd", "")
            if doc_name:
                aggregation[doc_name] = aggregation.get(doc_name, 0) + 1

        return self.SearchResult(total=total, ids=ids, query_vector=q_vec, aggregation=list(aggregation.items()), highlight=highlights, field=fields, keywords=list(kwds))

    @staticmethod
    def trans2floats(txt: str) -> List[float]:
        """Convert tab-separated string to float list"""
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer: str, chunks: List[str], chunk_v: List[List[float]], embd_mdl, tkweight: float = 0.1, vtweight: float = 0.9):
        """Insert citations into answer - same implementation as original"""
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set()

        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st:i]) + "\n")
                else:
                    pieces_.extend(re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)

        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]

        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)

        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set()

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split() for ck in chunks]
        cites: Dict[int, List[str]] = {}
        thr = 0.63

        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i], chunk_v, rag_tokenizer.tokenize(self.qryr.rmWWW(pieces_[i])).split(), chunks_tks, tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set()
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea: Optional[Dict], search_res: "QdrantDealer.SearchResult") -> np.ndarray:
        """Calculate rank feature scores - same implementation as original"""
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            if search_res.field:
                pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
            else:
                pageranks.append(0)
        pageranks_array = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks_array

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            if not search_res.field or not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        return np.array(rank_fea) * 10.0 + pageranks_array

    def rerank(self, sres: "QdrantDealer.SearchResult", query: str, tkweight: float = 0.3, vtweight: float = 0.7, cfield: str = "content_ltks", rank_feature: Optional[Dict] = None):
        """Rerank search results - same implementation as original"""
        _, keywords = self.qryr.question(query)
        if not sres.query_vector:
            return [], [], []

        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []

        if not sres.field:
            return [], [], []

        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)

        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]

        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(
        self, rerank_mdl, sres: "QdrantDealer.SearchResult", query: str, tkweight: float = 0.3, vtweight: float = 0.7, cfield: str = "content_ltks", rank_feature: Optional[Dict] = None
    ):
        """Rerank using external model - same implementation as original"""
        _, keywords = self.qryr.question(query)

        if not sres.field:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]

        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])

        # For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim) + rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans: str, inst: str):
        """Calculate hybrid similarity - same implementation as original"""
        return self.qryr.hybrid_similarity(ans_embd, ins_embd, rag_tokenizer.tokenize(ans).split(), rag_tokenizer.tokenize(inst).split())

    def retrieval(
        self,
        question: str,
        embd_mdl,
        tenant_ids: Union[str, List[str]],
        kb_ids: List[str],
        page: int,
        page_size: int,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        top: int = 1024,
        doc_ids: Optional[List[str]] = None,
        aggs: bool = True,
        rerank_mdl=None,
        highlight: bool = False,
        rank_feature: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Main retrieval method - same interface as original"""
        if rank_feature is None:
            rank_feature = {PAGERANK_FLD: 10}

        ranks: Dict[str, Any] = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        RERANK_LIMIT = 64
        RERANK_LIMIT = int(RERANK_LIMIT // page_size + ((RERANK_LIMIT % page_size) / (page_size * 1.0) + 0.5)) * page_size if page_size > 1 else 1
        if RERANK_LIMIT < 1:
            RERANK_LIMIT = 1

        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": math.ceil(page_size * page / RERANK_LIMIT),
            "size": RERANK_LIMIT,
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1,
        }

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        sres = self.search(req, [index_name(tid) for tid in tenant_ids], kb_ids, embd_mdl, highlight, rank_feature=rank_feature)

        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(rerank_mdl, sres, question, 1 - vector_similarity_weight, vector_similarity_weight, rank_feature=rank_feature)
        else:
            sim, tsim, vsim = self.rerank(sres, question, 1 - vector_similarity_weight, vector_similarity_weight, rank_feature=rank_feature)

        # Already paginated in search function
        idx = np.argsort(sim * -1)[(page - 1) * page_size : page * page_size]
        dim = len(sres.query_vector) if sres.query_vector else 0
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        sim_np = np.array(sim)

        if doc_ids:
            similarity_threshold = 0
        filtered_count = (sim_np >= similarity_threshold).sum()
        ranks["total"] = int(filtered_count)

        chunks_list = ranks["chunks"]
        doc_aggs = ranks["doc_aggs"]

        for i in idx:
            if sim[i] < similarity_threshold:
                break

            id = sres.ids[i]
            chunk = sres.field[id] if sres.field else {}
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            if len(chunks_list) >= page_size:
                if aggs:
                    if dnm not in doc_aggs:
                        doc_aggs[dnm] = {"doc_id": did, "count": 0}
                    doc_aggs[dnm]["count"] += 1
                    continue
                break

            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk.get("content_ltks", ""),
                "content_with_weight": chunk.get("content_with_weight", ""),
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk.get("kb_id", ""),
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }

            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]

            chunks_list.append(d)
            if dnm not in doc_aggs:
                doc_aggs[dnm] = {"doc_id": did, "count": 0}
            doc_aggs[dnm]["count"] += 1

        ranks["doc_aggs"] = [{"doc_name": k, "doc_id": v["doc_id"], "count": v["count"]} for k, v in sorted(doc_aggs.items(), key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = chunks_list[:page_size]

        return ranks

    def sql_retrieval(self, sql: str, fetch_size: int = 128, format: str = "json"):
        """SQL retrieval - not supported in Qdrant"""
        logging.warning("SQL retrieval not supported in Qdrant")
        return []

    def chunk_list(self, doc_id: str, tenant_id: str, kb_ids: List[str], max_count: int = 1024, offset: int = 0, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List chunks for a document"""
        if fields is None:
            fields = ["docnm_kwd", "content_with_weight", "img_id"]

        if not QDRANT_AVAILABLE:
            return []

        collection_name = index_name(tenant_id)

        try:
            filter_condition = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

            scroll_result = self.client.scroll(collection_name=collection_name, scroll_filter=filter_condition, limit=max_count, offset=offset, with_payload=True, with_vectors=False)

            res = []
            for point in scroll_result[0]:
                doc = {}
                for field in fields:
                    doc[field] = point.payload.get(field, "")
                doc["id"] = str(point.id)
                res.append(doc)

            return res

        except Exception as e:
            logging.error(f"Error listing chunks: {e}")
            return []

    def all_tags(self, tenant_id: str, kb_ids: List[str], S: int = 1000) -> List[tuple]:
        """Get all tags - simplified implementation"""
        if not QDRANT_AVAILABLE:
            return []

        collection_name = index_name(tenant_id)

        try:
            # This is a simplified implementation
            # In practice, you'd need to implement aggregation functionality
            scroll_result = self.client.scroll(collection_name=collection_name, limit=S, with_payload=True, with_vectors=False)

            tag_counts: Dict[str, int] = {}
            for point in scroll_result[0]:
                tags = point.payload.get("tag_kwd", [])
                if isinstance(tags, str):
                    tags = [tags]
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return list(tag_counts.items())

        except Exception as e:
            logging.error(f"Error getting all tags: {e}")
            return []

    def all_tags_in_portion(self, tenant_id: str, kb_ids: List[str], S: int = 1000) -> Dict[str, float]:
        """Get all tags with portions"""
        tags = self.all_tags(tenant_id, kb_ids, S)
        total = np.sum([c for _, c in tags])
        return {t: (c + 1) / (total + S) for t, c in tags}

    def tag_content(self, tenant_id: str, kb_ids: List[str], doc: Dict[str, Any], all_tags: Dict[str, float], topn_tags: int = 3, keywords_topn: int = 30, S: int = 1000) -> bool:
        """Tag content - simplified implementation"""
        # This would require implementing full-text search capabilities
        # For now, return False to indicate tagging was not performed
        logging.warning("Content tagging not fully implemented for Qdrant")
        return False

    def tag_query(self, question: str, tenant_ids: Union[str, List[str]], kb_ids: List[str], all_tags: Dict[str, float], topn_tags: int = 3, S: int = 1000) -> Dict[str, int]:
        """Tag query - simplified implementation"""
        # This would require implementing full-text search capabilities
        # For now, return empty dict
        logging.warning("Query tagging not fully implemented for Qdrant")
        return {}
