/**
 * @file history_tree.hpp
 * @author Jilles S. Dibangoye
 * @author David Albert
 * @brief History Tree data structure
 * @version 0.1
 * @date 14/12/2020
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once

#include <sdm/utils/struct/tree.hpp>
#include <sdm/core/state/history_tree_interface.hpp>

namespace sdm
{

    /**
     * @class HistoryTree
     * 
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class HistoryTree : public Tree<T>,
                        public BoostSerializable<HistoryTree<T>>, 
                        public HistoryTreeInterface
    {
    protected:
        /*!
         *  @brief  Expands the tree using truncated expand method
         *  @param  data the data of the expanded node
         *  @param  backup wheter the node is marked or not
         *  @return the truncated expanded tree
         */
        std::shared_ptr<HistoryTreeInterface> truncatedExpand(const std::shared_ptr<Observation> &observation, const std::shared_ptr<Action> &action, bool backup);

    public:
        using value_type = typename Tree<T>::value_type;
        /*!
         *  @brief  Default constructor.
         *  This constructor builds a default and empty tree.
         */
        HistoryTree();

        /**
         * @brief Construct a new truncated tree object
         * 
         * @param data the value of the origin 
         */
        HistoryTree(number max_depth);

        /*!
         *  @brief  constructor
         *  @param  parent   the parent tree
         *  @param  item     the item
         *  @param  backup wheter the node is marked or not
         *  This constructor builds a tree with a given parent and item.
         */
        HistoryTree(std::shared_ptr<HistoryTree<T>> parent, const T &item);

        /*!
         *  @brief  Expands the tree
         *  @param  data the data of the expanded node
         *  @return the expanded tree
         *
         *  If child leading from the item previously exists, the method return
         *  that child. Otherwise, it expands the tree by adding an item at the
         *  current leaf of the tree and creating if necessary a corresponding
         *  child. The constructed child is returned.
         */
        std::shared_ptr<HistoryTreeInterface> expand(const std::shared_ptr<Observation>&, const std::shared_ptr<Action>&, bool = true);

        /**
         * @brief Get the horizon
         * 
         * @return number 
         */
        number getHorizon() const;

        std::string str() const;
        std::string short_str() const;

        std::shared_ptr<HistoryTree<T>> getptr();

        template <class Archive>
        void serialize(Archive &archive, const unsigned int);

        std::shared_ptr<HistoryTree<T>> getParent() const;
        std::shared_ptr<HistoryTree<T>> getOrigin();
        std::vector<std::shared_ptr<HistoryTree<T>>> getChildren() const;
        std::shared_ptr<HistoryTree<T>> getChild(const T &child_item) const;

        friend std::ostream &operator<<(std::ostream &os, HistoryTree &i_hist)
        {
            os << i_hist.str();
            return os;
        }

        TypeState getTypeState() const{return TypeState::State_;}

    };

} // namespace sdm
#include <sdm/core/state/history_tree.tpp>
