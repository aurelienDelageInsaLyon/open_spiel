#include <sdm/utils/linear_algebra/hyperplane/vectorMG.hpp>

namespace sdm
{

    VectorMG::VectorMG(std::map<std::shared_ptr<HistoryInterface>, double> & values, double default_value, const std::shared_ptr<OccupancyStateMG> & state_support, int agent_id_) : Hyperplane(default_value), occ_support(state_support), agent_id_(agent_id_)
    {
        for (std::map<std::shared_ptr<HistoryInterface>, double>::iterator it = values.begin();it != values.end();it++){
            this->repr.emplace(it->first,it->second);
        }
    }
    
    std::shared_ptr<PrivateOccupancyState> VectorMG::getSigmaC(std::shared_ptr<HistoryInterface> history){
        
        //std::shared_ptr<PrivateBrOccupancyState> sigmaC1 = std::make_shared<PrivateBrOccupancyState>(state->getPrivateOccupancyState(this->agent_id_,history), this->agent_id_, this->);
        
        return this->occ_support->getPrivateOccupancyState(this->agent_id_,history);
    }
    double VectorMG::PRECISION = 0.00;

    double VectorMG::getInitValue(){
        return this->repr.begin()->second;
    }
    double VectorMG::getValueAt(const std::shared_ptr<HistoryInterface> &o) const
    {   
        //std::cout << "\n vector : " << this->str();
        //std::cout << "\n get value at o : " << o->short_str();
        auto iter = this->repr.find(o);
        if (iter == this->repr.end()){
            //std::cout << "\n returning default value :  " << this->default_value;
            return this->default_value;
        }
        else{
            return iter->second;
        }
    }
    void VectorMG::setValueAt(const std::shared_ptr<HistoryInterface> &o, double value)
    {

        auto iter = this->repr.find(o);
        if (iter == this->repr.end())
        {
            this->repr.emplace(o, default_value);
        }
        this->repr.at(o) = value;
    }

    double VectorMG::getValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o) const
     {
        throw sdm::exception::Exception("Bad call to getValueAt function in VectorMG. Vector only store values for histories");
     }
    void VectorMG::setValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o, double value)
    {
        throw sdm::exception::Exception("Bad call to setValueAt function in VectorMG. Vector only store values for histories");
    }

    double VectorMG::getValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o, const std::shared_ptr<Action>& u) const
     {
        throw sdm::exception::Exception("Bad call to getValueAt function in VectorMG. Vector only store values for histories");
     }
    void VectorMG::setValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o,const std::shared_ptr<Action>& u, double value)
    {
        throw sdm::exception::Exception("Bad call to setValueAt function in VectorMG. Vector only store values for histories");
    }

    bool VectorMG::isDominated(const Hyperplane &other) const
    {
        throw sdm::exception::Exception("Bad call to isDominated function in VectorMG.");
    }

    size_t VectorMG::hash(double precision) const
    {
        return 0;
    }

    bool VectorMG::isEqual(const VectorMG &other, double precision) const
    {

        if (precision < 0)
            precision = VectorMG::PRECISION;

        for (const auto &state_vector : this->repr)
        {
            if (std::abs(state_vector.second - other.getValueAt(state_vector.first, nullptr, nullptr)) > precision)
                    return false;
            
        }
        return true;
    }

    bool VectorMG::isEqual(const std::shared_ptr<Hyperplane> &other, double precision) const
    {
        auto other_oalpha = std::static_pointer_cast<VectorMG>(other);
        if (other_oalpha == nullptr)
            return false;
        else
            return true;
    }

    size_t VectorMG::size() const
    {
        return this->repr.size();
    }

    std::string VectorMG::str() const
    {
        std::ostringstream res;
        res << "<plan>" << std::endl;

        for (const auto &state_vector : this->repr)
        {
                res << "\t\t" << state_vector.first->str() << ", value : " << state_vector.second << std::endl;
            
        }
        for (auto & deltaStrategy : this->treeDeltaStrategies){
            if (deltaStrategy.second>0.0){
                if (std::get<2>(deltaStrategy.first)!=0){
                    res << deltaStrategy.second << " . <strategy> " << std::get<1>(deltaStrategy.first)->str() << "\t" << std::get<2>(deltaStrategy.first)->str();
                }
                else{
                    res << deltaStrategy.second << " . <strategy> " << std::get<1>(deltaStrategy.first)->str();
                }
            }
        }
        res << "\t</>" << std::endl;

        res << "</plan>" << std::endl;
        return res.str();
    }
} // namespace sdm
